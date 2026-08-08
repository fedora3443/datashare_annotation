package org.icij.datashare.llm;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * File-based lock manager to prevent concurrent access to shared resources (e.g., LLM service).
 * Uses file locking mechanism to ensure only one worker processes annotations at a time.
 */
public class FileLockManager implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(FileLockManager.class);
    
    private final Path lockFilePath;
    private RandomAccessFile lockFile;
    private FileChannel channel;
    private FileLock lock;
    
    private static final String DEFAULT_LOCK_FILE_NAME = "datashare-annotate.lock";
    private static final String DEFAULT_LOCK_DIR = System.getProperty("java.io.tmpdir");

    public FileLockManager() {
        this(Paths.get(DEFAULT_LOCK_DIR, DEFAULT_LOCK_FILE_NAME));
    }

    public FileLockManager(Path lockFilePath) {
        this.lockFilePath = lockFilePath;
    }

    public FileLockManager(String lockFilePath) {
        this(Paths.get(lockFilePath));
    }

    /**
     * Attempts to acquire an exclusive lock on the lock file.
     * 
     * @return true if lock was acquired, false if another process holds the lock
     * @throws IOException if an I/O error occurs
     */
    public boolean tryLock() throws IOException {
        try {
            // Ensure parent directory exists
            File parentDir = lockFilePath.getParent().toFile();
            if (!parentDir.exists()) {
                parentDir.mkdirs();
            }

            File file = lockFilePath.toFile();
            if (!file.exists()) {
                file.createNewFile();
            }

            lockFile = new RandomAccessFile(file, "rw");
            channel = lockFile.getChannel();
            
            // Try to acquire exclusive lock (non-blocking)
            lock = channel.tryLock();
            
            if (lock == null) {
                logger.info("Could not acquire lock on {}, another process is running", lockFilePath);
                close();
                return false;
            }
            
            logger.info("Successfully acquired lock on {}", lockFilePath);
            return true;
            
        } catch (Exception e) {
            logger.error("Error acquiring lock", e);
            close();
            throw e instanceof IOException ? (IOException) e : new IOException(e);
        }
    }

    /**
     * Acquires an exclusive lock, blocking until available.
     * 
     * @throws IOException if an I/O error occurs
     */
    public void lock() throws IOException {
        try {
            // Ensure parent directory exists
            File parentDir = lockFilePath.getParent().toFile();
            if (!parentDir.exists()) {
                parentDir.mkdirs();
            }

            File file = lockFilePath.toFile();
            if (!file.exists()) {
                file.createNewFile();
            }

            lockFile = new RandomAccessFile(file, "rw");
            channel = lockFile.getChannel();
            
            // Block until lock is acquired
            lock = channel.lock();
            
            logger.info("Acquired lock on {}", lockFilePath);
            
        } catch (Exception e) {
            logger.error("Error acquiring lock", e);
            close();
            throw e instanceof IOException ? (IOException) e : new IOException(e);
        }
    }

    /**
     * Releases the lock if held.
     */
    public void release() {
        try {
            if (lock != null && lock.isValid()) {
                lock.release();
                lock = null;
                logger.info("Released lock on {}", lockFilePath);
            }
        } catch (IOException e) {
            logger.error("Error releasing lock", e);
        } finally {
            close();
        }
    }

    @Override
    public void close() {
        try {
            if (channel != null && channel.isOpen()) {
                channel.close();
            }
        } catch (IOException e) {
            logger.error("Error closing channel", e);
        }
        
        try {
            if (lockFile != null) {
                lockFile.close();
            }
        } catch (IOException e) {
            logger.error("Error closing lock file", e);
        }
    }
}
