# Research Publication Template Structure Reference

## Core Files and Their Purposes

### Configuration Files
- **`.clinerules`**: Project-specific development rules and workflow guidelines
- **`_.code-workspace`**: VS Code workspace configuration with data directory settings
- **`gitignore`**: Git ignore patterns for research projects
- **`.env`**: Environment variables file (create from scratch as needed)

### Documentation Structure
- **`description/master.md`**: Research project requirements and definition
- **`memory-bank/`**: Technical documentation and research progress tracking
  - `project-brief.md`: Formalized research questions and hypotheses
  - `tech-stack.md`: Research tools, libraries, and technologies
  - `system-architecture.md`: Research methodology and structure
  - `implementation-plan.md`: Step-by-step research checklist
  - `progress.md`: Tracking of research progress

### Directory Structure
- **`src/`**: Source code implementations
  - `analysis/`: Data analysis scripts
  - `ai-assisted/`: AI-generated code (must be validated)
  - `models/`: Machine learning model implementations
  - `utils/`: Utility functions and helpers
- **`tests/`**: Validation tests for research implementations
- **`data/`**: Research data files
  - `raw/`: Original, unmodified data (read-only)
  - `processed/`: Cleaned and transformed data
- **`ref/`**: Reference materials and citations
  - `papers/`: Reference papers (PDFs or links)
  - `web-searches/`: Web search results and notes
  - `citations.bib`: Bibliography file
- **`papers/`**: Manuscript drafts and publication materials
- **`experiments/`**: Experimental setups and results
  - `README.md`: Experiment documentation and reproducibility instructions
- **`output/`**: Generated outputs (automatically created)
  - `XAI/SHAP/`: Example feature-specific output
    - `visualization/`: Figures and plots
    - `report/`: Markdown reports
  - `XAI/Grad-CAM/`: Another feature example
  - `ai/`: AI tool outputs

## Environment Variables

Key environment variables used in this template:

```bash
# Gemini API Configuration (if using AI tools)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-pro

# Data Directories
DATAFOLDER=/path/to/your/data/folder
DATA_DIR=${DATAFOLDER}/TestTemplate/data

# Research Configuration
RESEARCH_PROJECT_NAME=TestTemplate
RESEARCH_AUTHOR=Your Name
RESEARCH_INSTITUTION=Your Institution

# Output Configuration
OUTPUT_DIR=./output
```

## Common Workflow Patterns

### Starting a New Research Project
1. Create `.env` file with environment variables (see template above)
2. Read `description/master.md` for project requirements
3. Create `memory-bank/project-brief.md` from requirements
4. Set up `ref/` folder with initial literature
5. Create `src/` structure based on research needs

### Adding a New Analysis
1. Create script in `src/analysis/`
2. Use environment variables for data paths
3. Save outputs to appropriate `output/` subdirectory
4. Generate both visualizations and reports
5. Update `memory-bank/progress.md`

### Publishing Results
1. Write manuscript in `papers/` directory
2. Ensure all code is in `src/` and well-documented
3. Verify outputs are in `output/` with proper organization
4. Update references in `ref/citations.bib`
5. Run tests to ensure reproducibility