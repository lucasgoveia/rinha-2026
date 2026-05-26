const std = @import("std");
const linux = std.os.linux;
const posix = std.posix;
const math = std.math;

const MAX_BACKENDS: usize = 2;
const POOL_SIZE: usize = 64;
const MAX_RELAYS: usize = 128;
const SPLICE_SIZE: usize = 65536;
const EPOLL_MAX: usize = 256;

const SPLICE_F_MOVE: usize = 1;

// epoll userdata:
//   math.maxInt(u64)  → listen socket (accept)
//   slot | (1 << 7)   → backend fd event for relay[slot]
//   slot              → client fd event for relay[slot]
const UD_ACCEPT: u64 = math.maxInt(u64);

inline fn udClient(slot: u7) u64 {
    return @as(u64, slot);
}
inline fn udBackend(slot: u7) u64 {
    return @as(u64, slot) | (@as(u64, 1) << 7);
}
inline fn udIsBackend(ud: u64) bool {
    return ((ud >> 7) & 1) != 0;
}
inline fn udSlot(ud: u64) u7 {
    return @truncate(ud & 0x7F);
}

const Relay = struct {
    client_fd: i32 = -1,
    backend_fd: i32 = -1,
    backend_idx: u8 = 0,
    backend_slot: u6 = 0,
    pipe_c2b: [2]i32 = .{ -1, -1 }, // [0]=read [1]=write
    pipe_b2c: [2]i32 = .{ -1, -1 },
    active: bool = false,
};

var epoll_fd: i32 = -1;
var backend_fds: [MAX_BACKENDS][POOL_SIZE]i32 = undefined;
var backend_free: [MAX_BACKENDS]u64 = .{ math.maxInt(u64), math.maxInt(u64) };
var backend_paths: [MAX_BACKENDS][108]u8 = undefined;
var backend_path_lens: [MAX_BACKENDS]usize = .{ 0, 0 };
var n_backends: usize = 0;
var next_backend: u32 = 0;
var relays: [MAX_RELAYS]Relay = undefined;
var relay_free: u128 = math.maxInt(u128); // 1 = free

// ── slot helpers ─────────────────────────────────────────────────────────────

fn acquireBackendSlot(b: usize) ?u6 {
    if (backend_free[b] == 0) return null;
    const slot: u6 = @truncate(@ctz(backend_free[b]));
    backend_free[b] &= ~(@as(u64, 1) << slot);
    return slot;
}

fn releaseBackendSlot(b: usize, slot: u6) void {
    backend_free[b] |= @as(u64, 1) << slot;
}

fn acquireRelaySlot() ?u7 {
    if (relay_free == 0) return null;
    const slot: u7 = @truncate(@ctz(relay_free));
    relay_free &= ~(@as(u128, 1) << slot);
    return slot;
}

fn releaseRelaySlot(slot: u7) void {
    relay_free |= @as(u128, 1) << slot;
}

// ── syscall wrappers ─────────────────────────────────────────────────────────

fn doSplice(fd_in: i32, fd_out: i32, len: usize) isize {
    return @bitCast(linux.syscall6(
        .splice,
        @bitCast(@as(isize, fd_in)),
        0,
        @bitCast(@as(isize, fd_out)),
        0,
        len,
        SPLICE_F_MOVE,
    ));
}

// ── backend connection ───────────────────────────────────────────────────────

fn connectBackend(b: usize) i32 {
    const fd_rc = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.NONBLOCK, 0);
    const fd: i32 = @bitCast(@as(u32, @truncate(fd_rc)));
    if (fd < 0) return -1;

    var addr = std.mem.zeroes(linux.sockaddr.un);
    addr.family = linux.AF.UNIX;
    const len = backend_path_lens[b];
    @memcpy(addr.path[0..len], backend_paths[b][0..len]);

    const rc = linux.connect(fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.un));
    if (rc != 0) {
        const err = linux.errno(rc);
        if (err != .INPROGRESS and err != .AGAIN) {
            _ = linux.close(fd);
            return -1;
        }
    }
    return fd;
}

// ── epoll helpers ─────────────────────────────────────────────────────────────

fn epollAdd(fd: i32, events: u32, ud: u64) void {
    var ev = linux.epoll_event{ .events = events, .data = .{ .u64 = ud } };
    _ = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_ADD, fd, &ev);
}

fn epollDel(fd: i32) void {
    _ = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_DEL, fd, null);
}

// ── relay lifecycle ──────────────────────────────────────────────────────────

fn closeRelay(slot: u7, keep_backend: bool) void {
    const r = &relays[slot];
    if (!r.active) return;

    epollDel(r.client_fd);
    epollDel(r.backend_fd);
    _ = linux.close(r.client_fd);

    if (keep_backend) {
        // Return fd to pool for reuse.
        releaseBackendSlot(r.backend_idx, r.backend_slot);
    } else {
        // Backend fd may be unhealthy; close and reconnect into the pool slot.
        _ = linux.close(r.backend_fd);
        const new_fd = connectBackend(r.backend_idx);
        backend_fds[r.backend_idx][r.backend_slot] = new_fd;
        if (new_fd >= 0) releaseBackendSlot(r.backend_idx, r.backend_slot);
    }

    if (r.pipe_c2b[0] >= 0) {
        _ = linux.close(r.pipe_c2b[0]);
        _ = linux.close(r.pipe_c2b[1]);
    }
    if (r.pipe_b2c[0] >= 0) {
        _ = linux.close(r.pipe_b2c[0]);
        _ = linux.close(r.pipe_b2c[1]);
    }

    r.active = false;
    releaseRelaySlot(slot);
}

fn newRelay(client_fd: i32) void {
    const b = next_backend % n_backends;
    next_backend +%= 1;

    const bslot = acquireBackendSlot(b) orelse {
        _ = linux.close(client_fd);
        return;
    };

    var bfd = backend_fds[b][bslot];
    if (bfd < 0) {
        bfd = connectBackend(b);
        if (bfd < 0) {
            releaseBackendSlot(b, bslot);
            _ = linux.close(client_fd);
            return;
        }
        backend_fds[b][bslot] = bfd;
    }

    const rslot = acquireRelaySlot() orelse {
        releaseBackendSlot(b, bslot);
        _ = linux.close(client_fd);
        return;
    };

    var pipe_c2b: [2]i32 = undefined;
    var pipe_b2c: [2]i32 = undefined;
    if (linux.pipe2(&pipe_c2b, .{ .NONBLOCK = true }) != 0) {
        releaseBackendSlot(b, bslot);
        releaseRelaySlot(rslot);
        _ = linux.close(client_fd);
        return;
    }
    if (linux.pipe2(&pipe_b2c, .{ .NONBLOCK = true }) != 0) {
        _ = linux.close(pipe_c2b[0]);
        _ = linux.close(pipe_c2b[1]);
        releaseBackendSlot(b, bslot);
        releaseRelaySlot(rslot);
        _ = linux.close(client_fd);
        return;
    }

    relays[rslot] = Relay{
        .client_fd = client_fd,
        .backend_fd = bfd,
        .backend_idx = @truncate(b),
        .backend_slot = bslot,
        .pipe_c2b = pipe_c2b,
        .pipe_b2c = pipe_b2c,
        .active = true,
    };

    const ev_flags = linux.EPOLL.IN | linux.EPOLL.RDHUP | linux.EPOLL.ERR | linux.EPOLL.HUP;
    epollAdd(client_fd, ev_flags, udClient(rslot));
    epollAdd(bfd, ev_flags, udBackend(rslot));
}

// ── main ─────────────────────────────────────────────────────────────────────

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.skip(); // prog name
    var listen_port: u16 = 9999;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--listen")) {
            const a = it.next() orelse return error.MissingArg;
            const colon = std.mem.lastIndexOf(u8, a, ":") orelse return error.BadAddr;
            listen_port = try std.fmt.parseInt(u16, a[colon + 1 ..], 10);
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const path = it.next() orelse return error.MissingArg;
            if (n_backends >= MAX_BACKENDS) return error.TooManyBackends;
            const b = n_backends;
            n_backends += 1;
            backend_path_lens[b] = path.len;
            @memcpy(backend_paths[b][0..path.len], path);
            for (&backend_fds[b]) |*fd| fd.* = -1;
        }
    }

    if (n_backends == 0) return error.NoBackends;

    for (&relays) |*r| r.active = false;

    epoll_fd = @bitCast(@as(u32, @truncate(linux.epoll_create1(0))));
    if (epoll_fd < 0) return error.EpollCreate;

    const listen_fd_rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.NONBLOCK, 0);
    const listen_fd: i32 = @bitCast(@as(u32, @truncate(listen_fd_rc)));
    if (listen_fd < 0) return error.SocketFailed;

    var opt: i32 = 1;
    _ = linux.setsockopt(listen_fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, @ptrCast(&opt), @sizeOf(i32));

    var addr = std.mem.zeroes(linux.sockaddr.in);
    addr.family = linux.AF.INET;
    addr.port = std.mem.nativeToBig(u16, listen_port);
    addr.addr = 0;
    var rc = linux.bind(listen_fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in));
    if (rc != 0) return error.BindFailed;

    rc = linux.listen(listen_fd, 65535);
    if (rc != 0) return error.ListenFailed;

    epollAdd(listen_fd, linux.EPOLL.IN, UD_ACCEPT);

    // Pre-connect backend pool (best-effort; on-demand connect handles failures).
    for (0..n_backends) |b| {
        for (0..POOL_SIZE) |s| {
            backend_fds[b][s] = connectBackend(b);
            // Leave free-bit set regardless — newRelay reconnects if bfd==-1.
        }
    }

    var events: [EPOLL_MAX]linux.epoll_event = undefined;

    while (true) {
        const n = linux.epoll_wait(epoll_fd, &events, EPOLL_MAX, -1);
        for (events[0..n]) |ev| {
            const ud = ev.data.u64;
            const revents = ev.events;

            if (ud == UD_ACCEPT) {
                var client_addr: linux.sockaddr.in = undefined;
                var client_addrlen: posix.socklen_t = @sizeOf(linux.sockaddr.in);
                const crc = linux.accept4(listen_fd, @ptrCast(&client_addr), &client_addrlen, linux.SOCK.NONBLOCK);
                const cfd: i32 = @bitCast(@as(u32, @truncate(crc)));
                if (cfd >= 0) {
                    var nodelay: i32 = 1;
                    // IPPROTO_TCP=6, TCP_NODELAY=1
                    _ = linux.setsockopt(cfd, 6, 1, @ptrCast(&nodelay), @sizeOf(i32));
                    newRelay(cfd);
                }
                continue;
            }

            const slot = udSlot(ud);
            const r = &relays[slot];
            if (!r.active) continue;

            if (revents & (linux.EPOLL.ERR | linux.EPOLL.HUP) != 0) {
                closeRelay(slot, false);
                continue;
            }

            const is_backend = udIsBackend(ud);

            if (revents & linux.EPOLL.IN != 0) {
                if (!is_backend) {
                    // client → backend via pipe_c2b
                    const n_in = doSplice(r.client_fd, r.pipe_c2b[1], SPLICE_SIZE);
                    if (n_in > 0) {
                        _ = doSplice(r.pipe_c2b[0], r.backend_fd, @intCast(n_in));
                    } else {
                        closeRelay(slot, false);
                        continue;
                    }
                } else {
                    // backend → client via pipe_b2c
                    const n_in = doSplice(r.backend_fd, r.pipe_b2c[1], SPLICE_SIZE);
                    if (n_in > 0) {
                        _ = doSplice(r.pipe_b2c[0], r.client_fd, @intCast(n_in));
                    } else {
                        closeRelay(slot, false);
                        continue;
                    }
                }
            }

            if (revents & linux.EPOLL.RDHUP != 0) {
                // Remote side closed write end. Return backend to pool only if
                // client closed cleanly (backend may still be healthy).
                closeRelay(slot, !is_backend);
            }
        }
    }
}
