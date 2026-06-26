// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::fs::File;
use std::io;
use std::path::Path;

use paimon_mosaic_core::reader::InputFile;

/// A read-only [`InputFile`] backed by a real file using positional reads.
///
/// `read_exact_at` does not move a shared cursor, so concurrent calls from the
/// reader's coalescing threads are safe — satisfying the `Sync` bound.
pub struct FileInput {
    file: File,
    len: u64,
}

impl FileInput {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len();
        Ok(Self { file, len })
    }

    pub fn len(&self) -> u64 {
        self.len
    }
}

impl InputFile for FileInput {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            self.file.read_exact_at(buf, offset)
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::FileExt;
            let mut read = 0;
            while read < buf.len() {
                let n = self
                    .file
                    .seek_read(&mut buf[read..], offset + read as u64)?;
                if n == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "read past end",
                    ));
                }
                read += n;
            }
            Ok(())
        }
    }
}
