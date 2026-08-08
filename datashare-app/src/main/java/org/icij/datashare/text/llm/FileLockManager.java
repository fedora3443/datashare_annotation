package org.icij.datashare.text.llm;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.Paths;

/**
 * Manages file-based locking to prevent concurrent access to LLM resources.
 * Only one worker can hold the lock at a time, ensuring sequential processing.
 */
public class FileLockManager implements AutoCloseable {
    private static final Logger LOGGER = LoggerFactory.getLogger(FileLockManager.class);
    
    private final String lockFilePath;
    private FileOutputStream fos;
    private FileChannel channel;
    private FileLock lock;

    public FileLockManager(String lockFilePath) {
        this.lockFilePath = lockFilePath;
    }

    /**
     * Try to acquire an exclusive lock on the lock file.
     * 
     * @return true if lock was acquired successfully, false if another process holds it
     * @throws IOException If file operations fail
     */
    public boolean tryLock() throws IOException {
        File lockFile = new File(lockFilePath);
        
        // Ensure parent directory exists
        File parentDir = lockFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        fos = new FileOutputStream(lockFile);
        channel = fos.getChannel();
        
        try {
            lock = channel.tryLock();
            if (lock != null) {
                LOGGER.info("Acquired lock on {}", lockFilePath);
                return true;
            } else {
                LOGGER.debug("Could not acquire lock on {} - another process is running", lockFilePath);
                close();
                return false;
            }
        } catch (IOException e) {
            LOGGER.warn("Failed to acquire lock: {}", e.getMessage());
            close();
            throw e;
        }
    }

    /**
     * Acquire an exclusive lock, blocking until available.
     * 
     * @throws IOException If file operations fail or interrupted
     */
    public void lock() throws IOException {
        File lockFile = new File(lockFilePath);
        
        // Ensure parent directory exists
        File parentDir = lockFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        fos = new FileOutputStream(lockFile);
        channel = fos.getChannel();
        
        try {
            lock = channel.lock();
            LOGGER.info("Acquired exclusive lock on {}", lockFilePath);
        } catch (IOException | InterruptedException e) {
            LOGGER.error("Failed to acquire lock: {}", e.getMessage());
            Thread.currentThread().interrupt();
            close();
            throw new IOException("Lock acquisition failed", e);
        }
    }

    /**
     * Release the lock if held.
     */
    public void release() {
        if (lock != null && lock.isValid()) {
            try {
                lock.release();
                lock = null;
                LOGGER.info("Released lock on {}", lockFilePath);
            } catch (IOException e) {
                LOGGER.warn("Error releasing lock: {}", e.getMessage());
            }
        }
    }

    @Override
    public void close() {
        release();
        if (channel != null) {
            try {
                channel.close();
                channel = null;
            } catch (IOException e) {
                LOGGER.warn("Error closing channel: {}", e.getMessage());
            }
        }
        if (fos != null) {
            try {
                fos.close();
                fos = null;
            } catch (IOException e) {
                LOGGER.warn("Error closing stream: {}", e.getMessage());
            }
        }
        
        // Optionally delete the lock file after release
        File lockFile = new File(lockFilePath);
        if (lockFile.exists()) {
            if (lockFile.delete()) {
                LOGGER.debug("Deleted lock file {}", lockFilePath);
            } else {
                LOGGER.debug("Could not delete lock file {}", lockFilePath);
            }
        }
    }
}
