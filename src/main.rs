use std::sync::atomic::{AtomicPtr, Ordering};

use api_lib::ivf;
use api_lib::vectorize;

const BUF_SIZE: usize = 8192;
const MAX_CONNS: usize = 128;

const RESP_READY: &[u8] =
    b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";
const RESP_404: &[u8] =
    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

static FRAUD_RESPS: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\nConnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\nConnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

static IDX_PTR: AtomicPtr<ivf::IvfIndex> = AtomicPtr::new(std::ptr::null_mut());

struct Conn {
    fd: i32,
    buf: [u8; BUF_SIZE],
    buf_len: usize,
    write_resp: &'static [u8],
    write_pos: usize,
}

fn make_unix_listener(path: &str) -> i32 {
    unsafe {
        let fd = libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_NONBLOCK, 0);
        assert!(fd >= 0, "socket failed");
        let one: libc::c_int = 1;
        libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_REUSEADDR,
            &one as *const _ as *const libc::c_void, std::mem::size_of_val(&one) as u32);
        let mut addr: libc::sockaddr_un = std::mem::zeroed();
        addr.sun_family = libc::AF_UNIX as u16;
        let bytes = path.as_bytes();
        assert!(bytes.len() < addr.sun_path.len(), "sock path too long");
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr() as *const libc::c_char,
            addr.sun_path.as_mut_ptr(),
            bytes.len(),
        );
        let rc = libc::bind(fd, &addr as *const _ as *const libc::sockaddr,
            std::mem::size_of::<libc::sockaddr_un>() as u32);
        assert!(rc == 0, "bind failed");
        libc::chmod(path.as_ptr() as *const libc::c_char, 0o666);
        let rc = libc::listen(fd, 65535);
        assert!(rc == 0, "listen failed");
        fd
    }
}

// Receive a client TCP fd sent by the LB via SCM_RIGHTS.
// Returns -1 on EAGAIN or error (caller should break the drain loop).
// ctrl_fd must be nonblocking.
unsafe fn recv_fd(ctrl_fd: i32) -> i32 {
    let mut dummy = [0u8; 1];
    let mut iov = libc::iovec {
        iov_base: dummy.as_mut_ptr() as *mut libc::c_void,
        iov_len: 1,
    };
    // CMSG_SPACE(sizeof(int)) on Linux x86_64:
    //   cmsghdr = 16 bytes (len:8 + level:4 + type:4)
    //   CMSG_ALIGN(4) = 8 bytes for the fd
    //   total = 24 bytes
    let mut cmsg_buf = [0u8; 24];
    let mut msg: libc::msghdr = std::mem::zeroed();
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf.as_mut_ptr() as *mut libc::c_void;
    msg.msg_controllen = cmsg_buf.len() as _;
    let n = libc::recvmsg(ctrl_fd, &mut msg, libc::MSG_DONTWAIT);
    if n <= 0 { return -1; }
    let cmsg = libc::CMSG_FIRSTHDR(&msg);
    if cmsg.is_null() { return -1; }
    let cm = &*cmsg;
    if cm.cmsg_level != libc::SOL_SOCKET || cm.cmsg_type != libc::SCM_RIGHTS { return -1; }
    *(libc::CMSG_DATA(cmsg) as *const i32)
}

#[inline(always)]
fn find_crlfcrlf(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 { return None; }
    let mut i = 0;
    let end = buf.len() - 3;
    while i < end {
        if buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[inline(always)]
fn parse_content_length(headers: &[u8]) -> usize {
    let needle = b"Content-Length:";
    if headers.len() < needle.len() { return 0; }
    let mut i = 0;
    let end = headers.len() - needle.len();
    while i <= end {
        if headers[i] == b'C' && &headers[i..i + needle.len()] == needle {
            let mut j = i + needle.len();
            while j < headers.len() && (headers[j] == b' ' || headers[j] == b'\t') { j += 1; }
            let mut n = 0usize;
            while j < headers.len() {
                let b = headers[j];
                if b < b'0' || b > b'9' { break; }
                n = n * 10 + (b - b'0') as usize;
                j += 1;
            }
            return n;
        }
        i += 1;
    }
    0
}

#[inline(always)]
fn route(headers: &[u8], body: &[u8]) -> &'static [u8] {
    match headers.first().copied() {
        Some(b'P') => {
            if headers.starts_with(b"POST /fraud-score ") {
                let idx = unsafe { &*IDX_PTR.load(Ordering::Relaxed) };
                let vec = vectorize::parse_body(body);
                let fraud_count = idx.query(&vec).min(5) as usize;
                return unsafe { *FRAUD_RESPS.get_unchecked(fraud_count) };
            }
        }
        Some(b'G') => {
            if headers.starts_with(b"GET /ready ") { return RESP_READY; }
        }
        _ => {}
    }
    RESP_404
}

fn try_process_one(conn: &mut Conn) -> Option<&'static [u8]> {
    if conn.buf_len == 0 { return None; }
    let hdr_end = find_crlfcrlf(&conn.buf[..conn.buf_len])?;
    let content_length = parse_content_length(&conn.buf[..hdr_end]);
    let body_start = hdr_end + 4;
    let body_in_buf = conn.buf_len.saturating_sub(body_start);
    if body_in_buf < content_length { return None; }

    let resp = route(&conn.buf[..hdr_end], &conn.buf[body_start..body_start + content_length]);

    let next_start = body_start + content_length;
    conn.buf.copy_within(next_start..conn.buf_len, 0);
    conn.buf_len -= next_start;

    Some(resp)
}

fn try_sched_fifo() {
    unsafe {
        let mut p: libc::sched_param = std::mem::zeroed();
        p.sched_priority = 10;
        let rc = libc::sched_setscheduler(0, libc::SCHED_FIFO, &p);
        if rc != 0 {
            let e = *libc::__errno_location();
            eprintln!("[sched] SCHED_FIFO failed errno={} (continuing on SCHED_OTHER)", e);
        } else {
            eprintln!("[sched] SCHED_FIFO prio=10");
        }
    }
}

// Epoll userdata sentinels
const SLOT_ACCEPT: u64 = u64::MAX;
const SLOT_CTRL: u64 = u64::MAX - 1;

#[inline(always)]
fn epoll_mod(epfd: i32, fd: i32, events: u32, slot: u64) {
    let mut ev = libc::epoll_event { events, u64: slot };
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_MOD, fd, &mut ev) };
}

fn close_conn(slot: usize, conns: &mut [Option<Conn>; MAX_CONNS], epfd: i32) {
    if let Some(c) = conns[slot].take() {
        unsafe {
            libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, c.fd, std::ptr::null_mut());
            libc::close(c.fd);
        }
    }
}

fn add_client(fd: i32, conns: &mut [Option<Conn>; MAX_CONNS], epfd: i32) {
    match conns.iter().position(|c| c.is_none()) {
        None => unsafe { libc::close(fd); },
        Some(slot) => {
            conns[slot] = Some(Conn {
                fd,
                buf: [0; BUF_SIZE],
                buf_len: 0,
                write_resp: b"",
                write_pos: 0,
            });
            let mut ev = libc::epoll_event {
                events: libc::EPOLLIN as u32,
                u64: slot as u64,
            };
            unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
        }
    }
}

fn main() {
    let norm_path = std::env::var("NORM_PATH").unwrap_or_else(|_| "/data/normalization.json".to_string());
    let mcc_path = std::env::var("MCC_PATH").unwrap_or_else(|_| "/data/mcc_risk.json".to_string());
    vectorize::init(&norm_path, &mcc_path);

    let sock_path = std::env::var("SOCK").expect("SOCK env var required");
    let _ = std::fs::remove_file(&sock_path);

    let idx_path = std::env::var("INDEX_PATH").unwrap_or_else(|_| "/data/index.bin".to_string());
    let index = ivf::IvfIndex::load(&idx_path);
    let leaked: &'static mut ivf::IvfIndex = Box::leak(Box::new(index));
    IDX_PTR.store(leaked as *mut _, Ordering::Release);

    try_sched_fifo();

    // Unix socket acts as control channel: LB connects once and sends client fds via SCM_RIGHTS.
    let listen_fd = make_unix_listener(&sock_path);

    let epfd = unsafe { libc::epoll_create1(libc::EPOLL_CLOEXEC) };
    assert!(epfd >= 0, "epoll_create1 failed");

    let mut ev = libc::epoll_event { events: libc::EPOLLIN as u32, u64: SLOT_ACCEPT };
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, listen_fd, &mut ev) };

    let mut conns: Box<[Option<Conn>; MAX_CONNS]> =
        Box::new(std::array::from_fn(|_| None));

    let mut epevents = vec![libc::epoll_event { events: 0, u64: 0 }; 256];
    let mut batch: Vec<(u64, u32)> = Vec::with_capacity(256);

    // ctrl_fd: the single persistent control connection from the LB.
    let mut ctrl_fd: i32 = -1;

    loop {
        let n = unsafe {
            libc::epoll_wait(epfd, epevents.as_mut_ptr(), epevents.len() as i32, -1)
        };
        if n < 0 {
            let e = unsafe { *libc::__errno_location() };
            if e == libc::EINTR { continue; }
            break;
        }

        batch.clear();
        for i in 0..n as usize {
            batch.push((epevents[i].u64, epevents[i].events));
        }

        for &(ud, evflags) in &batch {
            // ── Accept LB control connection ──────────────────────────────────
            if ud == SLOT_ACCEPT {
                loop {
                    let fd = unsafe {
                        libc::accept4(listen_fd, std::ptr::null_mut(), std::ptr::null_mut(),
                            libc::SOCK_NONBLOCK)
                    };
                    if fd < 0 { break; }
                    // Close any stale control connection (LB restart).
                    if ctrl_fd >= 0 {
                        unsafe {
                            libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, ctrl_fd, std::ptr::null_mut());
                            libc::close(ctrl_fd);
                        }
                    }
                    ctrl_fd = fd;
                    let mut cev = libc::epoll_event {
                        events: (libc::EPOLLIN | libc::EPOLLRDHUP | libc::EPOLLERR | libc::EPOLLHUP) as u32,
                        u64: SLOT_CTRL,
                    };
                    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, ctrl_fd, &mut cev) };
                    eprintln!("[api] LB control connection established");
                }
                continue;
            }

            // ── Receive new client fds from LB ───────────────────────────────
            if ud == SLOT_CTRL {
                if evflags & (libc::EPOLLERR | libc::EPOLLHUP | libc::EPOLLRDHUP) as u32 != 0 {
                    eprintln!("[api] LB control connection lost");
                    unsafe {
                        libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, ctrl_fd, std::ptr::null_mut());
                        libc::close(ctrl_fd);
                    }
                    ctrl_fd = -1;
                    continue;
                }
                if evflags & libc::EPOLLIN as u32 != 0 {
                    loop {
                        let new_fd = unsafe { recv_fd(ctrl_fd) };
                        if new_fd < 0 { break; }
                        add_client(new_fd, &mut conns, epfd);
                    }
                }
                continue;
            }

            // ── Handle client connection ──────────────────────────────────────
            let slot = ud as usize;
            if conns[slot].is_none() { continue; }

            if evflags & (libc::EPOLLERR | libc::EPOLLHUP) as u32 != 0 {
                close_conn(slot, &mut conns, epfd);
                continue;
            }

            if evflags & libc::EPOLLIN as u32 != 0 {
                let conn = conns[slot].as_mut().unwrap();
                let space = BUF_SIZE - conn.buf_len;
                if space == 0 {
                    close_conn(slot, &mut conns, epfd);
                    continue;
                }
                let n = unsafe {
                    libc::recv(conn.fd, conn.buf.as_mut_ptr().add(conn.buf_len) as *mut libc::c_void, space, 0)
                };
                if n <= 0 {
                    close_conn(slot, &mut conns, epfd);
                    continue;
                }
                conn.buf_len += n as usize;

                if let Some(resp) = try_process_one(conn) {
                    conn.write_resp = resp;
                    conn.write_pos = 0;
                    let fd = conn.fd;
                    epoll_mod(epfd, fd, libc::EPOLLOUT as u32, slot as u64);
                }
            }

            if evflags & libc::EPOLLOUT as u32 != 0 {
                let conn = conns[slot].as_mut().unwrap();
                let remaining = &conn.write_resp[conn.write_pos..];
                let n = unsafe {
                    libc::send(conn.fd, remaining.as_ptr() as *const libc::c_void, remaining.len(), 0)
                };
                if n < 0 {
                    close_conn(slot, &mut conns, epfd);
                    continue;
                }
                conn.write_pos += n as usize;
                if conn.write_pos < conn.write_resp.len() {
                    continue;
                }

                let fd = conn.fd;
                if let Some(resp) = try_process_one(conn) {
                    conn.write_resp = resp;
                    conn.write_pos = 0;
                } else {
                    epoll_mod(epfd, fd, libc::EPOLLIN as u32, slot as u64);
                }
            }
        }
    }
}
