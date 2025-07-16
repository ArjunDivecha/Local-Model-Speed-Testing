# Requirements Document

## Introduction

This project combines the existing ModelSpeed.py and llm_evaluator.py programs into a single, streamlined LLM benchmarking and evaluation tool. The unified program will benchmark multiple LLM providers including OpenAI, Anthropic, Google, Groq, XAI, DeepSeek, and Cerebras, evaluate response quality using an AI judge, and generate focused visual and data outputs for analysis.

## Requirements

### Requirement 1

**User Story:** As a developer, I want to run a single command to benchmark and evaluate multiple LLM providers, so that I can efficiently compare models without managing multiple scripts and intermediate files.

#### Acceptance Criteria

1. WHEN the program is executed THEN it SHALL connect to all configured LLM providers with valid API keys
2. WHEN API keys are missing THEN the program SHALL skip those providers and continue with available ones
3. WHEN the program runs THEN it SHALL execute benchmarking and evaluation in a single process without creating intermediate HTML/TXT files
4. WHEN the program completes THEN it SHALL generate only the essential output files specified by the user

### Requirement 2

**User Story:** As a researcher, I want the program to test each model with the same prompt and measure both performance and quality metrics, so that I can make fair comparisons between different LLM providers.

#### Acceptance Criteria

1. WHEN testing models THEN the program SHALL use a consistent benchmark prompt across all providers
2. WHEN measuring performance THEN the program SHALL capture generation time, tokens per second, and time to first token
3. WHEN measuring quality THEN the program SHALL use Google Gemini as an AI judge to score responses on defined criteria
4. WHEN a model fails to respond THEN the program SHALL log the failure and continue with other models
5. WHEN calculating costs THEN the program SHALL use current provider pricing to estimate API usage costs

### Requirement 3

**User Story:** As an analyst, I want visual charts showing the relationship between speed and quality metrics, so that I can identify the best models for different use cases.

#### Acceptance Criteria

1. WHEN generating charts THEN the program SHALL create a Speed vs Quality scatter plot PDF
2. WHEN generating charts THEN the program SHALL create a Speed-only comparison chart PDF  
3. WHEN generating charts THEN the program SHALL create a Quality-only comparison chart PDF
4. WHEN generating charts THEN the program SHALL create a Time to First Token comparison chart PDF
5. WHEN creating charts THEN the program SHALL use consistent styling and clear labels
6. WHEN creating charts THEN the program SHALL highlight the reference model (Gemini) distinctly

### Requirement 4

**User Story:** As a data analyst, I want a comprehensive Excel file with all model metrics, so that I can perform custom analysis and create additional reports.

#### Acceptance Criteria

1. WHEN generating the Excel file THEN it SHALL include model name, provider, and platform information
2. WHEN generating the Excel file THEN it SHALL include performance metrics (generation time, tokens/sec, TTFT)
3. WHEN generating the Excel file THEN it SHALL include quality scores for each evaluation criterion
4. WHEN generating the Excel file THEN it SHALL include overall quality score and cost estimates
5. WHEN generating the Excel file THEN it SHALL use clear column headers and proper data formatting
6. WHEN generating the Excel file THEN it SHALL sort models by overall score in descending order

### Requirement 5

**User Story:** As a system administrator, I want the program to handle errors gracefully and provide clear feedback, so that I can troubleshoot issues and understand the results.

#### Acceptance Criteria

1. WHEN API calls fail THEN the program SHALL retry with exponential backoff up to 3 attempts
2. WHEN providers are unavailable THEN the program SHALL log clear error messages and continue
3. WHEN evaluation fails for a model THEN the program SHALL exclude it from reports but continue processing
4. WHEN the program starts THEN it SHALL validate API keys and report their status
5. WHEN the program completes THEN it SHALL provide a summary of successful and failed evaluations
6. WHEN generating outputs THEN the program SHALL confirm successful file creation with full paths

### Requirement 6

**User Story:** As a user, I want to configure which models and providers to test, so that I can focus on relevant comparisons and manage API costs.

#### Acceptance Criteria

1. WHEN configuring providers THEN the program SHALL read API keys from environment variables
2. WHEN configuring models THEN the program SHALL allow enabling/disabling specific providers
3. WHEN configuring testing THEN the program SHALL allow customization of the benchmark prompt
4. WHEN configuring testing THEN the program SHALL allow setting token limits for responses
5. WHEN configuring evaluation THEN the program SHALL allow customization of quality criteria