const std = @import("std");
const linux = std.os.linux;

const MAX_BACKENDS: usize = 2;

var ctrl_fds: [MAX_BACKENDS]i32 = .{ -1, -1 };
var backend_paths: [MAX_BACKENDS][108]u8 = undefined;
var backend_path_lens: [MAX_BACKENDS]usize = .{ 0, 0 };
var n_backends: usize = 0;
var next_backend: u32 = 0;

// Connect to backend's Unix control socket. Retries until the API is ready.
fn connectCtrl(b: usize) i32 {
    while (true) {
        const fd_rc = linux.socket(linux.AF.UNIX, linux.SOCK.STREAM, 0);
        const fd: i32 = @bitCast(@as(u32, @truncate(fd_rc)));
        if (fd < 0) {
            _ = linux.nanosleep(&.{ .sec = 0, .nsec = 10_000_000 }, null); // 10ms
            continue;
        }
        var addr = std.mem.zeroes(linux.sockaddr.un);
        addr.family = linux.AF.UNIX;
        const len = backend_path_lens[b];
        @memcpy(addr.path[0..len], backend_paths[b][0..len]);
        const rc = linux.connect(fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.un));
        if (rc == 0) return fd;
        const err = linux.errno(rc);
        _ = linux.close(fd);
        if (err == .CONNREFUSED or err == .NOENT or err == .AGAIN) {
            _ = linux.nanosleep(&.{ .sec = 0, .nsec = 10_000_000 }, null); // 10ms
            continue;
        }
        return -1;
    }
}

// Send client fd to backend via SCM_RIGHTS over the control socket.
// Uses raw byte buffers to match the kernel ABI on x86_64 Linux:
//   msghdr:   56 bytes (name:8 + namelen:4 + pad:4 + iov:8 + iovlen:8 + ctrl:8 + ctrllen:8 + flags:4 + pad:4)
//   cmsghdr:  16 bytes (len:8[size_t] + level:4 + type:4) + fd:4 + pad:4 = 24 bytes (CMSG_SPACE(4))
//   CMSG_LEN(4) = 16 + 4 = 20; CMSG_SPACE(4) = 16 + 8 = 24
fn sendFd(ctrl: i32, fd: i32) bool {
    var cmsg_buf: [24]u8 align(8) = std.mem.zeroes([24]u8);
    std.mem.writeInt(u64, cmsg_buf[0..8], 20, .little);       // cmsg_len (CMSG_LEN(4))
    std.mem.writeInt(i32, cmsg_buf[8..12], 1, .little);       // cmsg_level = SOL_SOCKET
    std.mem.writeInt(i32, cmsg_buf[12..16], 1, .little);      // cmsg_type = SCM_RIGHTS
    std.mem.writeInt(i32, cmsg_buf[16..20], fd, .little);     // payload: the fd

    var dummy: u8 = 0;
    // iovec on x86_64: base(8) + len(8) = 16 bytes
    var iov_buf: [16]u8 align(8) = std.mem.zeroes([16]u8);
    std.mem.writeInt(u64, iov_buf[0..8], @intFromPtr(&dummy), .little);
    std.mem.writeInt(u64, iov_buf[8..16], 1, .little);

    var msg_buf: [56]u8 align(8) = std.mem.zeroes([56]u8);
    // +16: msg_iov
    std.mem.writeInt(u64, msg_buf[16..24], @intFromPtr(&iov_buf), .little);
    // +24: msg_iovlen
    std.mem.writeInt(u64, msg_buf[24..32], 1, .little);
    // +32: msg_control
    std.mem.writeInt(u64, msg_buf[32..40], @intFromPtr(&cmsg_buf), .little);
    // +40: msg_controllen
    std.mem.writeInt(u64, msg_buf[40..48], cmsg_buf.len, .little);

    const rc = linux.syscall3(.sendmsg,
        @as(usize, @bitCast(@as(isize, ctrl))),
        @intFromPtr(&msg_buf),
        0,
    );
    return @as(isize, @bitCast(rc)) == 1;
}

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.skip();
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
        }
    }
    if (n_backends == 0) return error.NoBackends;

    // Connect control channels before accepting traffic (ensures APIs ready).
    for (0..n_backends) |b| {
        ctrl_fds[b] = connectCtrl(b);
        if (ctrl_fds[b] < 0) return error.BackendConnectFailed;
        std.debug.print("[lb] connected to backend {}\n", .{b});
    }

    // TCP listen socket (blocking accept is fine — we do no per-connection work).
    const listen_fd_rc = linux.socket(linux.AF.INET, linux.SOCK.STREAM, 0);
    const listen_fd: i32 = @bitCast(@as(u32, @truncate(listen_fd_rc)));
    if (listen_fd < 0) return error.SocketFailed;

    var opt: i32 = 1;
    _ = linux.setsockopt(listen_fd, linux.SOL.SOCKET, linux.SO.REUSEADDR, @ptrCast(&opt), @sizeOf(i32));
    _ = linux.setsockopt(listen_fd, linux.SOL.SOCKET, linux.SO.REUSEPORT, @ptrCast(&opt), @sizeOf(i32));

    var addr = std.mem.zeroes(linux.sockaddr.in);
    addr.family = linux.AF.INET;
    addr.port = std.mem.nativeToBig(u16, listen_port);
    addr.addr = 0;
    if (linux.bind(listen_fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in)) != 0) return error.BindFailed;
    if (linux.listen(listen_fd, 65535) != 0) return error.ListenFailed;

    std.debug.print("[lb] listening on :{}\n", .{listen_port});

    // Accept loop: receive TCP connection → forward fd to backend → close in LB.
    while (true) {
        const cfd_rc = linux.accept4(listen_fd, null, null, linux.SOCK.NONBLOCK);
        const cfd: i32 = @bitCast(@as(u32, @truncate(cfd_rc)));
        if (cfd < 0) continue;

        const b = next_backend % n_backends;
        next_backend +%= 1;
        _ = sendFd(ctrl_fds[b], cfd);
        _ = linux.close(cfd);
    }
}
