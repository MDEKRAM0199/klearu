//! Bridge between Ferret's `BufferIO` and klearu-mpc's `Transport` trait.

use ferret::BufferIO;
use klearu_mpc::Transport;
use std::io;

/// Pumps bytes between a [`BufferIO`] and a [`Transport`].
pub struct FerretTransport<T: Transport> {
    buffer_io: BufferIO,
    transport: T,
}

impl<T: Transport> FerretTransport<T> {
    pub fn new(buffer_io: BufferIO, transport: T) -> Self {
        Self { buffer_io, transport }
    }

    pub fn buffer_io(&self) -> &BufferIO {
        &self.buffer_io
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Drain the BufferIO send buffer into the transport,
    /// then fill the BufferIO receive buffer from the transport.
    pub fn flush(&mut self, expected_recv: usize) -> io::Result<()> {
        let send_size = self.buffer_io.send_size();
        if send_size > 0 {
            let data = self.buffer_io.drain_send(send_size);
            self.transport.send(&data)?;
        }
        if expected_recv > 0 {
            let data = self.transport.recv(expected_recv)?;
            self.buffer_io.fill_recv(&data).map_err(|e| {
                io::Error::new(io::ErrorKind::Other, e)
            })?;
        }
        Ok(())
    }
}
