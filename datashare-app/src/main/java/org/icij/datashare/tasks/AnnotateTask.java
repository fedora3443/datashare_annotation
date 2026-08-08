package org.icij.datashare.tasks;

import com.google.inject.Inject;
import com.google.inject.assistedinject.Assisted;
import org.icij.datashare.PropertiesProvider;
import org.icij.datashare.Stage;
import org.icij.datashare.asynctasks.Task;
import org.icij.datashare.asynctasks.TaskGroup;
import org.icij.datashare.asynctasks.TaskGroupType;
import org.icij.datashare.asynctasks.temporal.ActivityOpts;
import org.icij.datashare.asynctasks.temporal.TemporalSingleActivityWorkflow;
import org.icij.datashare.extract.DocumentCollectionFactory;
import org.icij.datashare.llm.FileLockManager;
import org.icij.datashare.llm.LlmClient;
import org.icij.datashare.llm.LlmClient.LlmAnnotationResult;
import org.icij.datashare.monitoring.Monitorable;
import org.icij.datashare.text.DocReference;
import org.icij.datashare.text.Document;
import org.icij.datashare.text.Project;
import org.icij.datashare.text.indexing.Indexer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import static java.util.Optional.ofNullable;
import static org.icij.datashare.PropertiesProvider.DEFAULT_PROJECT_OPT;
import static org.icij.datashare.cli.DatashareCliOptions.DEFAULT_DEFAULT_PROJECT;

/**
 * Task that annotates documents using an LLM and extracts emails/passwords.
 * Runs after indexing to process documents that don't have annotations yet.
 * Uses file-based locking to prevent concurrent access to the LLM service.
 */
@TemporalSingleActivityWorkflow(name = "annotate", activityOptions = @ActivityOpts(timeout = "PT60M"))
@TaskGroup(TaskGroupType.Java)
public class AnnotateTask extends PipelineTask<String> implements Monitorable {
    private static final Logger logger = LoggerFactory.getLogger(AnnotateTask.class);
    
    private static final int NB_MAX_POLLS = 3;
    private static final int POLLING_INTERVAL_SECONDS = 60;
    private static final int MAX_CONTENT_LENGTH = 64000; // Default max context length
    
    private final AtomicInteger processed = new AtomicInteger(0);
    private final Function<Double, Void> progressCallback;
    private final Indexer indexer;
    private final Project project;
    private final LlmClient llmClient;
    private final FileLockManager lockManager;
    private final String lockFilePath;
    
    @Inject
    public AnnotateTask(final Indexer indexer, 
                       final DocumentCollectionFactory<String> factory, 
                       @Assisted Task<Long> taskView, 
                       @Assisted final Function<Double, Void> progressCallback) {
        super(Stage.ANNOTATE, taskView.getUser(), factory, new PropertiesProvider(taskView.args), String.class);
        
        this.progressCallback = progressCallback;
        this.indexer = indexer;
        this.project = Project.project(
            ofNullable((String)taskView.args.get(DEFAULT_PROJECT_OPT))
                .orElse(DEFAULT_DEFAULT_PROJECT)
        );
        
        // Initialize LLM client from properties
        PropertiesProvider props = new PropertiesProvider(taskView.args);
        String llmBaseUrl = props.get("llm.baseUrl").orElse("http://localhost:8080");
        String llmApiKey = props.get("llm.apiKey").orElse("");
        String llmModel = props.get("llm.model").orElse(null);
        int maxContextLength = Integer.parseInt(
            props.get("llm.maxContextLength").orElse(String.valueOf(MAX_CONTENT_LENGTH))
        );
        this.lockFilePath = props.get("llm.lockFile")
            .orElse(System.getProperty("java.io.tmpdir") + "/datashare-annotate.lock");
        
        this.llmClient = new LlmClient(llmBaseUrl, llmApiKey, llmModel, maxContextLength);
        this.lockManager = new FileLockManager(lockFilePath);
        
        logger.info("AnnotateTask initialized with LLM URL: {}, model: {}, maxContext: {}", 
            llmBaseUrl, llmModel, maxContextLength);
    }

    @Override
    public Long call() throws Exception {
        super.call();
        
        logger.info("Starting annotation task for project {} with {} documents in queue", 
            project.getName(), inputQueue.size());
        
        // Try to acquire lock (non-blocking)
        if (!lockManager.tryLock()) {
            logger.warn("Could not acquire annotation lock at {}. Another instance is running. Exiting.", lockFilePath);
            return 0L;
        }
        
        try {
            return processDocuments();
        } finally {
            lockManager.release();
            logger.info("Released annotation lock");
        }
    }
    
    private Long processDocuments() throws Exception {
        String queueEntry;
        long nbMessages = 0;
        int nbMaxPolls = NB_MAX_POLLS;
        int pollingIntervalSeconds = POLLING_INTERVAL_SECONDS;

        while (!(STRING_POISON.equals(queueEntry = inputQueue.poll((pollingIntervalSeconds * 1000L), TimeUnit.MILLISECONDS)))
                && nbMaxPolls > 0) {
            try {
                if (queueEntry != null) {
                    Document retrievedFromIndexer = getDocument(indexer, project.getName(), DocReference.parse(queueEntry));
                    if (retrievedFromIndexer != null) {
                        annotateDocument(retrievedFromIndexer);
                    }
                    nbMessages++;
                    processed.incrementAndGet();
                    progressCallback.apply(getProgressRate());
                    
                    if (!outputQueue.offer(queueEntry)) {
                        logger.warn("Unable to offer {} to queue {}", queueEntry, outputQueue.getName());
                    }
                } else {
                    logger.info("Will poll document queue again for pollingInterval={} seconds ({}/{})", 
                        pollingIntervalSeconds, nbMaxPolls, NB_MAX_POLLS);
                    nbMaxPolls--;
                }
            } catch (Exception e) {
                logger.error("Error in AnnotateTask loop", e);
            }
        }
        
        if (!outputQueue.offer(STRING_POISON)) {
            logger.warn("Unable to offer POISON to queue {}", outputQueue.getName());
        }
        
        logger.info("Exiting AnnotateTask loop after {} messages.", nbMessages);
        return nbMessages;
    }

    private void annotateDocument(Document doc) throws IOException {
        // Check if already annotated
        Map<String, Object> metadata = doc.getMetadata();
        if (metadata != null && metadata.containsKey("annotation") && metadata.get("annotation") != null) {
            logger.debug("Document {} already has annotation, skipping", doc.getId());
            return;
        }
        
        String content = doc.getContent();
        if (content == null || content.trim().isEmpty()) {
            logger.debug("Document {} has no content, skipping annotation", doc.getId());
            return;
        }
        
        logger.info("Annotating document {}", doc.getId());
        
        try {
            LlmAnnotationResult result = llmClient.annotate(content);
            
            // Update document metadata with annotation results
            Map<String, Object> updatedMetadata = new HashMap<>(metadata != null ? metadata : Collections.emptyMap());
            updatedMetadata.put("annotation", result.getAnnotation());
            updatedMetadata.put("annotation_mails", result.getEmails());
            updatedMetadata.put("annotation_passwords", result.getPasswords());
            
            // Update the document in Elasticsearch
            indexer.update(project.getName(), doc.getId(), updatedMetadata);
            
            logger.info("Successfully annotated document {}: {} emails, {} passwords found", 
                doc.getId(), result.getEmails().size(), result.getPasswords().size());
                
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.error("Annotation interrupted for document {}", doc.getId(), e);
            throw new IOException("Annotation interrupted", e);
        } catch (Exception e) {
            logger.error("Failed to annotate document {}", doc.getId(), e);
            // Don't rethrow - continue processing other documents
        }
    }

    @Override
    public double getProgressRate() {
        int done = processed.get();
        int totalToProcess = done + inputQueue.size();
        return totalToProcess == 0 ? 0 : (double) done / totalToProcess;
    }
}
