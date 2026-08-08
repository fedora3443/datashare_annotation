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
 * Client for interacting with OpenAI-compatible LLM APIs (e.g., llama.cpp server).
 * Provides annotation generation and extraction of emails/passwords from text.
 */
public class LlmClient {
    private static final Logger logger = LoggerFactory.getLogger(LlmClient.class);
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    
    private final String baseUrl;
    private final String apiKey;
    private final String model;
    private final int maxContextLength;
    private final HttpClient httpClient;
    
    // Regex patterns for email and password extraction as fallback
    private static final Pattern EMAIL_PATTERN = Pattern.compile(
        "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}"
    );
    private static final Pattern PASSWORD_PATTERN = Pattern.compile(
        "(?i)(?:password|passwd|pwd|pass)\\s*[=:]\\s*['\"]?([^\\s'\"]+)['\"]?"
    );

    public LlmClient(String baseUrl, String apiKey, String model, int maxContextLength) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.apiKey = apiKey;
        this.model = model != null && !model.isEmpty() ? model : "default";
        this.maxContextLength = maxContextLength > 0 ? maxContextLength : 64000;
        this.httpClient = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();
    }

    /**
     * Generates an annotation for the given text and extracts emails and passwords.
     * 
     * @param content The text content to annotate
     * @return LlmAnnotationResult containing annotation, emails, and passwords
     * @throws IOException if the API call fails
     * @throws InterruptedException if the request is interrupted
     */
    public LlmAnnotationResult annotate(String content) throws IOException, InterruptedException {
        if (content == null || content.trim().isEmpty()) {
            return new LlmAnnotationResult("", Collections.emptyList(), Collections.emptyList());
        }

        // Truncate content if it exceeds max context length
        String truncatedContent = truncateContent(content, maxContextLength);
        
        String prompt = buildPrompt(truncatedContent);
        
        try {
            String response = callChatCompletion(prompt);
            return parseResponse(response);
        } catch (Exception e) {
            logger.error("LLM API call failed, falling back to regex extraction", e);
            // Fallback to regex-based extraction
            return extractWithRegex(truncatedContent);
        }
    }

    private String buildPrompt(String content) {
        return String.format(
            "You are a document analysis assistant. Analyze the following text and provide:\n" +
            "1. A concise annotation/summary in Russian (2-3 sentences)\n" +
            "2. All email addresses found\n" +
            "3. All passwords or credential-like strings found\n\n" +
            "Return ONLY a valid JSON object with this exact structure:\n" +
            "{\n" +
            "  \"annotation\": \"your summary in Russian\",\n" +
            "  \"emails\": [\"email1@example.com\", \"email2@test.org\"],\n" +
            "  \"passwords\": [\"password1\", \"secret123\"]\n" +
            "}\n\n" +
            "If no emails or passwords are found, use empty arrays.\n\n" +
            "TEXT TO ANALYZE:\n%s",
            content
        );
    }

    private String truncateContent(String content, int maxLength) {
        if (content.length() <= maxLength) {
            return content;
        }
        // Truncate and add indicator
        return content.substring(0, maxLength) + "\n...[truncated]";
    }

    private String callChatCompletion(String prompt) throws IOException, InterruptedException {
        String endpoint = baseUrl + "/v1/chat/completions";
        
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("model", model);
        requestBody.put("temperature", 0.1); // Low temperature for consistent output
        requestBody.put("max_tokens", 1000);
        
        List<Map<String, String>> messages = new ArrayList<>();
        Map<String, String> userMessage = new HashMap<>();
        userMessage.put("role", "user");
        userMessage.put("content", prompt);
        messages.add(userMessage);
        requestBody.put("messages", messages);

        String jsonBody = OBJECT_MAPPER.writeValueAsString(requestBody);
        
        HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
            .uri(URI.create(endpoint))
            .timeout(Duration.ofMinutes(2))
            .header("Content-Type", "application/json")
            .POST(HttpRequest.BodyPublishers.ofString(jsonBody));

        // Add API key if provided
        if (apiKey != null && !apiKey.isEmpty()) {
            requestBuilder.header("Authorization", "Bearer " + apiKey);
        }

        HttpRequest request = requestBuilder.build();
        
        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        
        if (response.statusCode() != 200) {
            throw new IOException("LLM API returned status code: " + response.statusCode() + 
                ", body: " + response.body());
        }

        return response.body();
    }

    private LlmAnnotationResult parseResponse(String jsonResponse) throws IOException {
        JsonNode root = OBJECT_MAPPER.readTree(jsonResponse);
        JsonNode choices = root.get("choices");
        
        if (choices == null || choices.isEmpty()) {
            throw new IOException("No choices in LLM response");
        }
        
        String content = choices.get(0).get("message").get("content").asText();
        
        // Try to extract JSON from the response (in case there's extra text)
        String jsonStr = extractJsonFromResponse(content);
        JsonNode resultNode = OBJECT_MAPPER.readTree(jsonStr);
        
        String annotation = resultNode.has("annotation") ? 
            resultNode.get("annotation").asText("") : "";
        
        List<String> emails = new ArrayList<>();
        if (resultNode.has("emails")) {
            JsonNode emailsNode = resultNode.get("emails");
            if (emailsNode.isArray()) {
                for (JsonNode emailNode : emailsNode) {
                    String email = emailNode.asText();
                    if (!email.isEmpty() && isValidEmail(email)) {
                        emails.add(email);
                    }
                }
            }
        }
        
        List<String> passwords = new ArrayList<>();
        if (resultNode.has("passwords")) {
            JsonNode passwordsNode = resultNode.get("passwords");
            if (passwordsNode.isArray()) {
                for (JsonNode pwdNode : passwordsNode) {
                    String pwd = pwdNode.asText();
                    if (!pwd.isEmpty()) {
                        passwords.add(pwd);
                    }
                }
            }
        }
        
        logger.info("Successfully parsed LLM response: {} emails, {} passwords", 
            emails.size(), passwords.size());
        
        return new LlmAnnotationResult(annotation, emails, passwords);
    }

    private String extractJsonFromResponse(String content) {
        // Try to find JSON object in the response
        int startIdx = content.indexOf("{");
        int endIdx = content.lastIndexOf("}");
        
        if (startIdx >= 0 && endIdx > startIdx) {
            return content.substring(startIdx, endIdx + 1);
        }
        return content;
    }

    private boolean isValidEmail(String email) {
        return EMAIL_PATTERN.matcher(email).matches();
    }

    private LlmAnnotationResult extractWithRegex(String content) {
        Set<String> emails = new LinkedHashSet<>();
        Set<String> passwords = new LinkedHashSet<>();
        
        // Extract emails
        Matcher emailMatcher = EMAIL_PATTERN.matcher(content);
        while (emailMatcher.find()) {
            emails.add(emailMatcher.group());
        }
        
        // Extract passwords
        Matcher pwdMatcher = PASSWORD_PATTERN.matcher(content);
        while (pwdMatcher.find()) {
            String pwd = pwdMatcher.group(1);
            if (pwd != null && !pwd.isEmpty()) {
                passwords.add(pwd);
            }
        }
        
        String fallbackAnnotation = "Автоматическая аннотация не удалась. Текст содержит " + 
            emails.size() + " email(s) и " + passwords.size() + " потенциальных паролей.";
        
        return new LlmAnnotationResult(fallbackAnnotation, 
            new ArrayList<>(emails), new ArrayList<>(passwords));
    }

    /**
     * Result class containing annotation, emails, and passwords.
     */
    public static class LlmAnnotationResult {
        private final String annotation;
        private final List<String> emails;
        private final List<String> passwords;

        public LlmAnnotationResult(String annotation, List<String> emails, List<String> passwords) {
            this.annotation = annotation != null ? annotation : "";
            this.emails = emails != null ? emails : Collections.emptyList();
            this.passwords = passwords != null ? passwords : Collections.emptyList();
        }

        public String getAnnotation() {
            return annotation;
        }

        public List<String> getEmails() {
            return Collections.unmodifiableList(emails);
        }

        public List<String> getPasswords() {
            return Collections.unmodifiableList(passwords);
        }
    }
}
