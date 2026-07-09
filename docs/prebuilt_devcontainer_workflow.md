# Pre-Built Dev Container Workflow

This document outlines the architecture and workflow for pre-building our development containers using Continuous Integration (CI). 

## Architecture Overview

Instead of building the Dev Container Docker image locally on developers' machines every time they open the project, we offload this heavy computational process to GitHub Actions.

1. **Trigger:** Whenever a developer pushes changes to the `main` branch that affect the environment (e.g., modifying `Dockerfile`, `devcontainer.json`, or `requirements.txt`), a GitHub Actions workflow is triggered.
2. **Cloud Build:** A GitHub-hosted runner executes the build process using the official `@devcontainers/ci` action. This ensures the environment is built exactly according to our specification.
3. **Artifact Storage:** The resulting Docker image is pushed to the GitHub Container Registry (GHCR).
4. **Local Consumption:** When a developer opens the project in their local IDE, the Dev Container configuration instructs Docker to *pull* the latest image from GHCR rather than building it locally.

## Benefits

*   **Drastically Reduced Startup Times:** "Cold starts" take seconds (just downloading the image layers) instead of minutes (installing OS packages and Python dependencies).
*   **Consistency:** Every developer is guaranteed to use the exact same pre-compiled environment image.
*   **Visibility:** Build history, cache hits, and performance metrics are visible in the GitHub Actions dashboard.
*   **Reduced Local Resource Usage:** Laptops are spared the CPU and battery drain of heavy Docker builds.

## Workflow Execution

### 1. The GitHub Actions Pipeline
The pipeline (`.github/workflows/build-devcontainer.yml`) performs the following:
* Checks out the code.
* Logs into GitHub Container Registry (GHCR) using the `GITHUB_TOKEN`.
* Runs the `@devcontainers/ci` action to build the image.
* Pushes the image tagged with `:latest` and a unique commit SHA tag.

### 2. Local Dev Container Configuration
The `.devcontainer/docker-compose.yml` file is configured so that the `app` service points to the GHCR image:

```yaml
services:
  app:
    image: ghcr.io/<org-name>/<repo-name>-devcontainer:latest
    # ... volume and environment configuration ...
```

### 3. Making Environment Changes
When a developer needs to install a new dependency or change the OS environment:
1. They modify `requirements.txt` or `.devcontainer/Dockerfile`.
2. They commit and push these changes in a Pull Request.
3. Once merged to `main`, the CI pipeline builds the new image.
4. Other developers pull the `main` branch, and their IDE automatically detects the new image tag and downloads it.

*(Note: During active testing of a new dependency before pushing, developers can temporarily revert the `image` flag to a `build` block in their local `docker-compose.yml` if they need to test the build locally before committing).*
