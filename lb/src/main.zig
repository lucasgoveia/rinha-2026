const std = @import("std");
const linux = std.os.linux;
const math = std.math;

const MAX_BACKENDS: usize = 2;
const POOL_SIZE: usize = 64;
const MAX_RELAYS: usize = MAX_BACKENDS * POOL_SIZE;
const EPOLL_BATCH: usize = 512;
const SPLICE_SIZE: usize = 65536;
const LISTEN_SENTINEL: u64 = 0xFFFF;

const SPLICE_F_MOVE: u32 = 1;
const SPLICE_F_NONBLOCK: u32 = 2;

const RelaySlot = struct {
    client_fd: i32 = -1,
    backend_idx: u8 = 0,
    backend_slot: u8 = 0,
    pipe_c2b: [2]i32 = .{ -1, -1 },
    pipe_b2c: [2]i32 = .{ -1, -1 },
    pending_c2b: u32 = 0,
    pending_b2c: u32 = 0,
    active: bool = false,
};

var backend_fds: [MAX_BACKENDS][POOL_SIZE]i32 = undefined;
var backend_free: [MAX_BACKENDS]u64 = .{ math.maxInt(u64), math.maxInt(u64) };
var backend_path_buf: [MAX_BACKENDS][108]u8 = undefined;
var backend_path_len: [MAX_BACKENDS]usize = .{ 0, 0 };
var relay_slots: [MAX_RELAYS]RelaySlot = undefined;
var relay_free: u128 = math.maxInt(u128);
var next_backend: u32 = 0;
var epoll_fd: i32 = -1;
var n_backends: usize = 0;

// ── slot helpers ─────────────────────────────────────────────────────────────

fn acquireSlot64(free: *u64) ?u6 {
    if (free.* == 0) return null;
    const idx: u6 = @truncate(@ctz(free.*));
    free.* &= ~(@as(u64, 1) << idx);
    return idx;
}

fn releaseSlot64(free: *u64, idx: u6) void {
    free.* |= @as(u64, 1) << idx;
}

fn acquireSlot128(free: *u128) ?u7 {
    if (free.* == 0) return null;
    const idx: u7 = @truncate(@ctz(free.*));
    free.* &= ~(@as(u128, 1) << idx);
    return idx;
}

fn releaseSlot128(free: *u128, idx: u7) void {
    free.* |= @as(u128, 1) << idx;
}

// ── syscall helpers ───────────────────────────────────────────────────────────

fn sysClose(fd: i32) void {
    _ = linux.close(@intCast(fd));
}

fn sysSetNonBlock(fd: i32) void {
    const flags = linux.fcntl(fd, linux.F.GETFL, 0);
    _ = linux.fcntl(fd, linux.F.SETFL, flags | linux.SOCK.NONBLOCK);
}

fn sysSplice(fd_in: i32, fd_out: i32, len: usize, flags: u32) isize {
    return @bitCast(linux.syscall6(
        .splice,
        @bitCast(@as(isize, fd_in)),
        0,
        @bitCast(@as(isize, fd_out)),
        0,
        len,
        flags,
    ));
}

// ── address parsing ───────────────────────────────────────────────────────────

fn parseIpv4(host: []const u8, port: u16) !linux.sockaddr.in {
    var octets: [4]u8 = undefined;
    var it = std.mem.splitScalar(u8, host, '.');
    for (&octets) |*o| {
        const part = it.next() orelse return error.InvalidAddress;
        o.* = try std.fmt.parseInt(u8, part, 10);
    }
    if (it.next() != null) return error.InvalidAddress;
    var addr: linux.sockaddr.in = std.mem.zeroes(linux.sockaddr.in);
    addr.port = std.mem.nativeToBig(u16, port);
    addr.addr = @bitCast(octets);
    return addr;
}

// ── backend connection ────────────────────────────────────────────────────────

fn connectBackend(b: usize) !i32 {
    const fd_rc = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM | linux.SOCK.NONBLOCK, 0);
    const fd: i32 = @bitCast(@as(u32, @truncate(fd_rc)));
    if (fd < 0) return error.SocketFailed;

    const path = backend_path_buf[b][0..backend_path_len[b]];
    var addr: linux.sockaddr.un = std.mem.zeroes(linux.sockaddr.un);
    addr.family = linux.AF.UNIX;
    @memcpy(addr.path[0..path.len], path);

    const rc = linux.connect(fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.un));
    const err = linux.errno(rc);
    if (rc != 0 and err != .INPROGRESS and err != .AGAIN) {
        sysClose(fd);
        return error.ConnectFailed;
    }
    return fd;
}

// ── epoll helpers ─────────────────────────────────────────────────────────────

fn epollAdd(fd: i32, events: u32, data: u64) !void {
    var ev = linux.epoll_event{
        .events = events,
        .data = .{ .u64 = data },
    };
    const rc = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_ADD, fd, &ev);
    if (rc != 0) return error.EpollCtlFailed;
}

fn epollDel(fd: i32) void {
    _ = linux.epoll_ctl(epoll_fd, linux.EPOLL.CTL_DEL, fd, null);
}

fn encodeEvent(slot: u7, dir: u1) u64 {
    return @as(u64, slot) | (@as(u64, dir) << 7);
}

// ── pipe drain (flush leftover data before reuse) ─────────────────────────────

fn drainPipe(pipe_r: i32) void {
    var buf: [4096]u8 = undefined;
    while (true) {
        const rc = linux.read(pipe_r, &buf, buf.len);
        const n: isize = @bitCast(rc);
        if (n <= 0) break;
    }
}

// ── relay lifecycle ───────────────────────────────────────────────────────────

fn initRelay(client_fd: i32, b: usize, bslot: u6, ridx: u7) !void {
    const s = &relay_slots[ridx];
    s.client_fd = client_fd;
    s.backend_idx = @intCast(b);
    s.backend_slot = bslot;
    s.pending_c2b = 0;
    s.pending_b2c = 0;
    s.active = true;

    drainPipe(s.pipe_c2b[0]);
    drainPipe(s.pipe_b2c[0]);

    const backend_fd = backend_fds[b][bslot];
    try epollAdd(client_fd, linux.EPOLL.IN | linux.EPOLL.HUP | linux.EPOLL.ERR | linux.EPOLL.ET, encodeEvent(ridx, 0));
    try epollAdd(backend_fd, linux.EPOLL.IN | linux.EPOLL.OUT | linux.EPOLL.HUP | linux.EPOLL.ERR | linux.EPOLL.ET, encodeEvent(ridx, 1));
}

fn closeRelay(ridx: u7) void {
    const s = &relay_slots[ridx];
    if (!s.active) return;
    s.active = false;

    epollDel(s.client_fd);
    sysClose(s.client_fd);
    s.client_fd = -1;

    const b: usize = s.backend_idx;
    const bslot: u6 = @truncate(s.backend_slot);
    const old_fd = backend_fds[b][bslot];
    epollDel(old_fd);
    sysClose(old_fd);

    backend_fds[b][bslot] = connectBackend(b) catch -1;
    releaseSlot64(&backend_free[b], bslot);
    releaseSlot128(&relay_free, ridx);
}

// ── splice relay ──────────────────────────────────────────────────────────────

const SpliceResult = union(enum) { ok: u32, eof, again };

fn spliceRelay(src_fd: i32, pipe_r: i32, pipe_w: i32, dst_fd: i32) SpliceResult {
    const n = sysSplice(src_fd, pipe_w, SPLICE_SIZE, SPLICE_F_MOVE | SPLICE_F_NONBLOCK);
    if (n == 0) return .eof;
    if (n < 0) {
        const err = linux.errno(@bitCast(n));
        if (err == .AGAIN) return .again;
        return .eof;
    }
    const nbytes: usize = @intCast(n);
    const drained = sysSplice(pipe_r, dst_fd, nbytes, SPLICE_F_MOVE | SPLICE_F_NONBLOCK);
    if (drained < 0) {
        const err = linux.errno(@bitCast(drained));
        if (err == .AGAIN) return .{ .ok = @intCast(nbytes) };
        return .eof;
    }
    const remaining = nbytes - @as(usize, @intCast(drained));
    return .{ .ok = @intCast(remaining) };
}

fn drainToDst(pipe_r: i32, dst_fd: i32, pending: u32) u32 {
    const n = sysSplice(pipe_r, dst_fd, pending, SPLICE_F_MOVE | SPLICE_F_NONBLOCK);
    if (n <= 0) return pending;
    const moved: u32 = @intCast(n);
    return if (moved >= pending) 0 else pending - moved;
}

// ── event handlers ────────────────────────────────────────────────────────────

fn handleClientEvent(ridx: u7, events: u32) void {
    const s = &relay_slots[ridx];
    if (!s.active) return;
    const backend_fd = backend_fds[s.backend_idx][s.backend_slot];

    // drain pending b2c pipe → client (client writable)
    if (events & linux.EPOLL.OUT != 0 and s.pending_b2c > 0) {
        s.pending_b2c = drainToDst(s.pipe_b2c[0], s.client_fd, s.pending_b2c);
    }
    // read client → backend (data available)
    if (events & linux.EPOLL.IN != 0 and s.pending_c2b == 0) {
        switch (spliceRelay(s.client_fd, s.pipe_c2b[0], s.pipe_c2b[1], backend_fd)) {
            .eof => { closeRelay(ridx); return; },
            .again => {},
            .ok => |p| s.pending_c2b = p,
        }
    }
    // handle HUP/ERR after draining any readable data
    if (events & (linux.EPOLL.HUP | linux.EPOLL.ERR) != 0) {
        if (events & linux.EPOLL.IN == 0) closeRelay(ridx);
    }
}

fn handleBackendEvent(ridx: u7, events: u32) void {
    const s = &relay_slots[ridx];
    if (!s.active) return;
    const backend_fd = backend_fds[s.backend_idx][s.backend_slot];

    // drain pending c2b pipe → backend (backend writable)
    if (events & linux.EPOLL.OUT != 0 and s.pending_c2b > 0) {
        s.pending_c2b = drainToDst(s.pipe_c2b[0], backend_fd, s.pending_c2b);
    }
    // read backend → client (data available) — check even when HUP to drain last bytes
    if (events & linux.EPOLL.IN != 0 and s.pending_b2c == 0) {
        switch (spliceRelay(backend_fd, s.pipe_b2c[0], s.pipe_b2c[1], s.client_fd)) {
            .eof => { closeRelay(ridx); return; },
            .again => {},
            .ok => |p| s.pending_b2c = p,
        }
    }
    // handle HUP/ERR after draining any readable data
    if (events & (linux.EPOLL.HUP | linux.EPOLL.ERR) != 0) {
        if (events & linux.EPOLL.IN == 0) closeRelay(ridx);
    }
}

// ── accept loop ───────────────────────────────────────────────────────────────

fn drainAccept(listen_fd: i32) void {
    while (true) {
        const rc = linux.accept4(listen_fd, null, null, linux.SOCK.NONBLOCK);
        const client_fd: i32 = @bitCast(@as(u32, @truncate(rc)));
        if (client_fd < 0) {
            const err = linux.errno(rc);
            if (err == .AGAIN) return;
            return;
        }

        const b: usize = blk: {
            const v = @atomicRmw(u32, &next_backend, .Add, 1, .monotonic);
            break :blk v % n_backends;
        };

        const bslot = acquireSlot64(&backend_free[b]) orelse {
            sysClose(client_fd);
            continue;
        };

        const ridx = acquireSlot128(&relay_free) orelse {
            releaseSlot64(&backend_free[b], bslot);
            sysClose(client_fd);
            continue;
        };

        initRelay(client_fd, b, bslot, ridx) catch {
            releaseSlot64(&backend_free[b], bslot);
            releaseSlot128(&relay_free, ridx);
            sysClose(client_fd);
        };
    }
}

// ── init ──────────────────────────────────────────────────────────────────────

fn initPipes() !void {
    const o_flags = linux.O{ .NONBLOCK = true, .CLOEXEC = true };
    for (0..MAX_RELAYS) |i| {
        var fds: [2]i32 = undefined;
        var rc = linux.pipe2(@ptrCast(&fds), o_flags);
        if (rc != 0) return error.PipeFailed;
        relay_slots[i].pipe_c2b = fds;

        rc = linux.pipe2(@ptrCast(&fds), o_flags);
        if (rc != 0) return error.PipeFailed;
        relay_slots[i].pipe_b2c = fds;

        relay_slots[i].active = false;
        relay_slots[i].client_fd = -1;
    }
}

fn initPool() !void {
    for (0..n_backends) |b| {
        for (0..POOL_SIZE) |s| {
            backend_fds[b][s] = try connectBackend(b);
        }
    }
}

fn setSockOpt(fd: i32, level: u32, optname: u32, val: i32) void {
    _ = linux.syscall5(
        .setsockopt,
        @bitCast(@as(isize, fd)),
        level,
        optname,
        @intFromPtr(&val),
        @sizeOf(i32),
    );
}

// ── arg parsing ───────────────────────────────────────────────────────────────

fn parseArgs(it: *std.process.Args.Iterator) !struct { host: []const u8, port: u16 } {
    var listen_host: []const u8 = "0.0.0.0";
    var listen_port: u16 = 9999;

    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--listen")) {
            const val = it.next() orelse return error.MissingArg;
            if (std.mem.indexOfScalar(u8, val, ':')) |colon| {
                listen_host = val[0..colon];
                listen_port = try std.fmt.parseInt(u16, val[colon + 1 ..], 10);
            } else {
                listen_port = try std.fmt.parseInt(u16, val, 10);
            }
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const val = it.next() orelse return error.MissingArg;
            if (n_backends >= MAX_BACKENDS) return error.TooManyBackends;
            const b = n_backends;
            n_backends += 1;
            const len = @min(val.len, backend_path_buf[b].len - 1);
            @memcpy(backend_path_buf[b][0..len], val[0..len]);
            backend_path_buf[b][len] = 0;
            backend_path_len[b] = len;
        }
    }

    return .{ .host = listen_host, .port = listen_port };
}

// ── main ──────────────────────────────────────────────────────────────────────

pub fn main(io: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(io.args);
    _ = it.next(); // skip argv[0]
    const listen_info = try parseArgs(&it);

    if (n_backends == 0) {
        _ = linux.write(2, "usage: lb --listen HOST:PORT --backend /path/to.sock\n", 52);
        return error.NoBackends;
    }

    try initPipes();
    try initPool();

    // listen socket
    const lfd_rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM | linux.SOCK.NONBLOCK, 0);
    const listen_fd: i32 = @bitCast(@as(u32, @truncate(lfd_rc)));
    if (listen_fd < 0) return error.SocketFailed;

    setSockOpt(listen_fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, 1);
    setSockOpt(listen_fd, linux.SOL.SOCKET, linux.SO.REUSEPORT, 1);

    const bind_addr = try parseIpv4(listen_info.host, listen_info.port);
    var rc = linux.bind(listen_fd, @ptrCast(&bind_addr), @sizeOf(linux.sockaddr.in));
    if (rc != 0) return error.BindFailed;

    rc = linux.listen(listen_fd, 65535);
    if (rc != 0) return error.ListenFailed;

    const efd_rc = linux.epoll_create1(linux.EPOLL.CLOEXEC);
    epoll_fd = @bitCast(@as(u32, @truncate(efd_rc)));
    if (epoll_fd < 0) return error.EpollFailed;

    try epollAdd(listen_fd, linux.EPOLL.IN | linux.EPOLL.ET, LISTEN_SENTINEL);

    var events: [EPOLL_BATCH]linux.epoll_event = undefined;
    while (true) {
        const n = linux.epoll_wait(epoll_fd, &events, EPOLL_BATCH, -1);
        for (events[0..n]) |ev| {
            if (ev.data.u64 == LISTEN_SENTINEL) {
                drainAccept(listen_fd);
            } else {
                const ridx: u7 = @truncate(ev.data.u64 & 0x7F);
                const dir: u1 = @truncate((ev.data.u64 >> 7) & 1);
                if (dir == 0) {
                    handleClientEvent(ridx, ev.events);
                } else {
                    handleBackendEvent(ridx, ev.events);
                }
            }
        }
    }
}
