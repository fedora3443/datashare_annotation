package org.icij.datashare.llm;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Client for OpenAI-compatible LLM APIs (e.g., llama.cpp server with Qwen model).
 * Sends document content to LLM and extracts annotation, emails, and passwords.
 */
public class LlmClient {
    private static final Logger LOGGER = LoggerFactory.getLogger(LlmClient.class);
    
    private final String baseUrl;
    private final String apiKey;
    private final String model;
    private final int maxContextLength;
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    
    // Regex patterns for fallback extraction
    private static final Pattern EMAIL_PATTERN = Pattern.compile(
        "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    );
    private static final Pattern PASSWORD_PATTERN = Pattern.compile(
        "(?i)(?:password|passwd|pwd|pass)\\s*[:=]\\s*[\"']?([a-zA-Z0-9@#$%^&*!_+\\-]{6,})[\"']?"
    );

    public LlmClient(String baseUrl, String apiKey, String model, int maxContextLength) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.apiKey = apiKey;
        this.model = model;
        this.maxContextLength = maxContextLength;
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Send document content to LLM and get annotation with extracted entities.
     * 
     * @param content Document content to analyze
     * @return LlmResult with annotation, emails, and passwords
     * @throws IOException If request fails
     * @throws InterruptedException If request is interrupted
     */
    public LlmResult analyze(String content) throws IOException, InterruptedException {
        if (content == null || content.trim().isEmpty()) {
            return new LlmResult("", Collections.emptyList(), Collections.emptyList());
        }

        // Truncate content if it exceeds max context length
        String truncatedContent = content.length() > maxContextLength 
            ? content.substring(0, maxContextLength) 
            : content;

        String prompt = buildPrompt(truncatedContent);
        
        try {
            String response = sendRequest(prompt);
            return parseResponse(response);
        } catch (Exception e) {
            LOGGER.warn("LLM request failed, using fallback extraction: {}", e.getMessage());
            return extractWithFallback(truncatedContent);
        }
    }

    private String buildPrompt(String content) {
        return String.format(
            "You are a document analysis assistant. Analyze the following document text and extract:\n" +
            "1. A brief annotation/summary (2-3 sentences in English)\n" +
            "2. All email addresses found\n" +
            "3. All passwords or credentials found\n\n" +
            "Return ONLY a valid JSON object with this exact structure:\n" +
            "{\n" +
            "  \"annotation\": \"your summary here\",\n" +
            "  \"emails\": [\"email1@example.com\", \"email2@test.org\"],\n" +
            "  \"passwords\": [\"password123\", \"secret456\"]\n" +
            "}\n\n" +
            "If no emails or passwords are found, return empty arrays.\n\n" +
            "Document text:\n" +
            "---BEGIN DOCUMENT---\n%s\n---END DOCUMENT---",
            content
        );
    }

    private String sendRequest(String prompt) throws IOException, InterruptedException {
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", model);
        requestBody.put("messages", List.of(
            Map.of("role", "user", "content", prompt)
        ));
        requestBody.put("temperature", 0.1);
        requestBody.put("max_tokens", 500);
        requestBody.put("response_format", Map.of("type", "json_object"));

        String jsonBody = objectMapper.writeValueAsString(requestBody);

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + "/v1/chat/completions"))
            .timeout(Duration.ofSeconds(120))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer " + (apiKey != null ? apiKey : "not-needed"))
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody))
            .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() != 200) {
            throw new IOException("LLM API returned status code: " + response.statusCode());
        }

        JsonNode rootNode = objectMapper.readTree(response.body());
        return rootNode.path("choices").get(0).path("message").path("content").asText();
    }

    private LlmResult parseResponse(String jsonResponse) throws IOException {
        JsonNode node = objectMapper.readTree(jsonResponse);
        
        String annotation = node.path("annotation").asText("");
        
        List<String> emails = new ArrayList<>();
        JsonNode emailsNode = node.path("emails");
        if (emailsNode.isArray()) {
            for (JsonNode emailNode : emailsNode) {
                String email = emailNode.asText();
                if (!email.isEmpty() && isValidEmail(email)) {
                    emails.add(email);
                }
            }
        }
        
        List<String> passwords = new ArrayList<>();
        JsonNode passwordsNode = node.path("passwords");
        if (passwordsNode.isArray()) {
            for (JsonNode pwdNode : passwordsNode) {
                String pwd = pwdNode.asText();
                if (!pwd.isEmpty()) {
                    passwords.add(pwd);
                }
            }
        }
        
        return new LlmResult(annotation, emails, passwords);
    }

    private LlmResult extractWithFallback(String content) {
        List<String> emails = new ArrayList<>();
        Matcher emailMatcher = EMAIL_PATTERN.matcher(content);
        while (emailMatcher.find()) {
            String email = emailMatcher.group();
            if (isValidEmail(email) && !emails.contains(email)) {
                emails.add(email);
            }
        }

        List<String> passwords = new ArrayList<>();
        Matcher pwdMatcher = PASSWORD_PATTERN.matcher(content);
        while (pwdMatcher.find()) {
            String pwd = pwdMatcher.group(1);
            if (pwd != null && !pwd.isEmpty() && !passwords.contains(pwd)) {
                passwords.add(pwd);
            }
        }

        // Generate a simple fallback annotation
        String annotation = String.format(
            "Document analyzed with fallback extraction. Found %d email(s) and %d potential password(s).",
            emails.size(), passwords.size()
        );

        return new LlmResult(annotation, emails, passwords);
    }

    private boolean isValidEmail(String email) {
        return email != null && EMAIL_PATTERN.matcher(email).matches();
    }

    /**
     * Result of LLM analysis containing annotation and extracted entities.
     */
    public static class LlmResult {
        private final String annotation;
        private final List<String> emails;
        private final List<String> passwords;

        public LlmResult(String annotation, List<String> emails, List<String> passwords) {
            this.annotation = annotation;
            this.emails = Collections.unmodifiableList(new ArrayList<>(emails));
            this.passwords = Collections.unmodifiableList(new ArrayList<>(passwords));
        }

        public String getAnnotation() {
            return annotation;
        }

        public List<String> getEmails() {
            return emails;
        }

        public List<String> getPasswords() {
            return passwords;
        }
    }
}
