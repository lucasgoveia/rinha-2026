use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

use api_lib::ivf;
use api_lib::vectorize;

const BUF_SIZE: usize = 8192;
const MAX_CONNS: usize = 128;
const EPOLL_BATCH: usize = 128;
const LISTEN_TOKEN: u64 = u64::MAX;

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

fn epoll_add(epfd: i32, fd: i32, token: u64, flags: u32) {
    let mut ev = libc::epoll_event { events: flags, u64: token };
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_ADD, fd, &mut ev) };
}

fn epoll_mod(epfd: i32, fd: i32, token: u64, flags: u32) {
    let mut ev = libc::epoll_event { events: flags, u64: token };
    unsafe { libc::epoll_ctl(epfd, libc::EPOLL_CTL_MOD, fd, &mut ev) };
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

// Fast byte-by-byte CRLFCRLF scan (no windows().position() overhead).
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
        // Case-insensitive on first char (C vs c) is fine — top compatible clients send Content-Length
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
    if headers.starts_with(b"GET /ready ") { return RESP_READY; }
    if headers.starts_with(b"POST /fraud-score ") {
        let idx = unsafe { &*IDX_PTR.load(Ordering::Relaxed) };
        let vec = vectorize::parse_body(body);
        let fraud_count = idx.query(&vec);
        return FRAUD_RESPS[fraud_count.min(5) as usize];
    }
    RESP_404
}

fn flush_write(conn: &mut Conn) -> bool {
    while conn.write_pos < conn.write_resp.len() {
        let n = unsafe {
            libc::send(
                conn.fd,
                conn.write_resp[conn.write_pos..].as_ptr() as *const libc::c_void,
                conn.write_resp.len() - conn.write_pos,
                libc::MSG_NOSIGNAL,
            )
        };
        if n < 0 {
            let e = unsafe { *libc::__errno_location() };
            if e == libc::EAGAIN || e == libc::EWOULDBLOCK { return true; }
            return false;
        }
        conn.write_pos += n as usize;
    }
    true
}

enum ProcResult { Processed, NeedMore, WriteBlocked, Error }

fn try_process_one(conn: &mut Conn, slot: usize, epfd: i32) -> ProcResult {
    if conn.write_pos < conn.write_resp.len() { return ProcResult::WriteBlocked; }
    if conn.buf_len == 0 { return ProcResult::NeedMore; }

    let Some(hdr_end) = find_crlfcrlf(&conn.buf[..conn.buf_len]) else {
        return ProcResult::NeedMore;
    };

    let content_length = parse_content_length(&conn.buf[..hdr_end]);
    let body_start = hdr_end + 4;
    let body_in_buf = conn.buf_len.saturating_sub(body_start);

    if body_in_buf < content_length {
        return ProcResult::NeedMore;
    }

    let resp = route(&conn.buf[..hdr_end], &conn.buf[body_start..body_start + content_length]);
    conn.write_resp = resp;
    conn.write_pos = 0;

    let next_start = body_start + content_length;
    conn.buf.copy_within(next_start..conn.buf_len, 0);
    conn.buf_len -= next_start;

    if !flush_write(conn) { return ProcResult::Error; }

    if conn.write_pos < conn.write_resp.len() {
        epoll_mod(epfd, conn.fd, slot as u64,
            libc::EPOLLIN as u32 | libc::EPOLLOUT as u32 | libc::EPOLLET as u32);
        return ProcResult::WriteBlocked;
    }

    ProcResult::Processed
}

fn process_loop(slot: usize, epfd: i32, conns: &mut Vec<Option<Conn>>) -> bool {
    loop {
        let conn = match conns[slot].as_mut() { Some(c) => c, None => return false };
        match try_process_one(conn, slot, epfd) {
            ProcResult::Processed => continue,
            ProcResult::NeedMore | ProcResult::WriteBlocked => return true,
            ProcResult::Error => return false,
        }
    }
}

fn close_conn(slot: usize, epfd: i32, conns: &mut Vec<Option<Conn>>) {
    if let Some(c) = conns[slot].take() {
        unsafe {
            libc::epoll_ctl(epfd, libc::EPOLL_CTL_DEL, c.fd, null_mut());
            libc::close(c.fd);
        }
    }
}

fn drain_accept(listen_fd: i32, epfd: i32, conns: &mut Vec<Option<Conn>>) {
    loop {
        let fd = unsafe { libc::accept4(listen_fd, null_mut(), null_mut(), libc::SOCK_NONBLOCK) };
        if fd < 0 { break; }
        match conns.iter().position(|c| c.is_none()) {
            None => unsafe { libc::close(fd); },
            Some(i) => {
                conns[i] = Some(Conn {
                    fd,
                    buf: [0; BUF_SIZE],
                    buf_len: 0,
                    write_resp: b"",
                    write_pos: 0,
                });
                epoll_add(epfd, fd, i as u64, libc::EPOLLIN as u32 | libc::EPOLLET as u32);
            }
        }
    }
}

fn handle_conn(slot: usize, events: u32, epfd: i32, conns: &mut Vec<Option<Conn>>) {
    if events & libc::EPOLLOUT as u32 != 0 {
        {
            let conn = match conns[slot].as_mut() { Some(c) => c, None => return };
            if !flush_write(conn) { close_conn(slot, epfd, conns); return; }
            if conn.write_pos < conn.write_resp.len() { return; }
            let fd = conn.fd;
            epoll_mod(epfd, fd, slot as u64, libc::EPOLLIN as u32 | libc::EPOLLET as u32);
        }
        if !process_loop(slot, epfd, conns) { close_conn(slot, epfd, conns); return; }
        if conns[slot].is_none() { return; }
    }

    if events & libc::EPOLLIN as u32 != 0 {
        loop {
            let conn = match conns[slot].as_mut() { Some(c) => c, None => return };
            let space = BUF_SIZE - conn.buf_len;
            if space == 0 { close_conn(slot, epfd, conns); return; }
            let n = unsafe {
                libc::recv(conn.fd, conn.buf[conn.buf_len..].as_mut_ptr() as *mut libc::c_void, space, 0)
            };
            if n == 0 { close_conn(slot, epfd, conns); return; }
            if n < 0 {
                let e = unsafe { *libc::__errno_location() };
                if e == libc::EAGAIN || e == libc::EWOULDBLOCK { break; }
                close_conn(slot, epfd, conns); return;
            }
            conns[slot].as_mut().unwrap().buf_len += n as usize;
            if !process_loop(slot, epfd, conns) { close_conn(slot, epfd, conns); return; }
            if conns[slot].is_none() { return; }
        }
    }

    if events & (libc::EPOLLHUP as u32 | libc::EPOLLERR as u32) != 0 {
        if events & libc::EPOLLIN as u32 == 0 { close_conn(slot, epfd, conns); }
    }
}

fn try_sched_fifo() {
    unsafe {
        let mut p: libc::sched_param = std::mem::zeroed();
        p.sched_priority = 10;
        let rc = libc::sched_setscheduler(0, libc::SCHED_FIFO, &p);
        if rc != 0 {
            let e = *libc::__errno_location();
            eprintln!("[sched] sched_setscheduler SCHED_FIFO failed errno={} (continuing on SCHED_OTHER)", e);
        } else {
            eprintln!("[sched] SCHED_FIFO prio=10");
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

    let listen_fd = make_unix_listener(&sock_path);

    let epfd = unsafe { libc::epoll_create1(0) };
    assert!(epfd >= 0);
    epoll_add(epfd, listen_fd, LISTEN_TOKEN, libc::EPOLLIN as u32 | libc::EPOLLET as u32);

    let mut conns: Vec<Option<Conn>> = (0..MAX_CONNS).map(|_| None).collect();
    let mut events: Vec<libc::epoll_event> =
        (0..EPOLL_BATCH).map(|_| libc::epoll_event { events: 0, u64: 0 }).collect();

    loop {
        let n = unsafe { libc::epoll_wait(epfd, events.as_mut_ptr(), EPOLL_BATCH as i32, -1) };
        if n < 0 { continue; }
        for i in 0..n as usize {
            let token = events[i].u64;
            let evflags = events[i].events;
            if token == LISTEN_TOKEN {
                drain_accept(listen_fd, epfd, &mut conns);
            } else {
                handle_conn(token as usize, evflags, epfd, &mut conns);
            }
        }
    }
}
