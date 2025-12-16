# Animal Genetics AI Personalized Learning App

## 1. Project Overview

Welcome to the Animal Genetics AI Personalized Learning App, a state-of-the-art educational tool designed for undergraduate students in Animal Science. This application provides a comprehensive, interactive, and personalized learning experience based on the textbook "Animal Genetics" (Li Ning, 3rd Edition) and the latest university curriculum standards.

Our mission is to transform traditional learning by leveraging AI, knowledge graphs, and interactive simulations to create a "one-student, one-solution" educational model. The app is built on Outcome-Based Education (OBE) principles, focusing on mastering core concepts and applying them in practical scenarios.

### Key Features:

*   **AI-Powered Personalization**: A dynamic learning path that adapts to each student's pace and mastery level.
*   **Interactive Knowledge Graph**: Visualize the entire curriculum, explore connections between concepts, and navigate your learning journey.
*   **Intelligent Tutoring**: An AI-powered tutor available 24/7 to answer questions, explain complex topics, and provide step-by-step guidance on problem-solving.
*   **Virtual Laboratory**: Conduct simulated genetics experiments, from Mendelian crosses to population genetics modeling, in a safe and interactive environment.
*   **Real-World Case Studies**: Apply your knowledge to solve practical problems in animal breeding and genetic improvement.
*   **Adaptive Assessments**: Quizzes and tests that adjust in difficulty based on your performance, providing instant feedback and diagnostics.
*   **Gamification & Motivation**: Earn points, badges, and achievements to stay motivated. Compete with classmates on leaderboards.
*   **Bilingual Support**: Seamlessly switch between English and Chinese to support diverse learning environments.
*   **Teacher & Platform Integration**: Full support for teachers to manage courses, track student progress, and integrate with the Chaoxing Learning Platform.

---

## 2. User Guide

This section provides a guide for the two main user groups: Students and Teachers.

### 2.1 For Students

As a student, this app is your personal guide to mastering animal genetics.

#### Getting Started
1.  **Login**: Use your university credentials or Chaoxing account to log in.
2.  **Initial Assessment**: On your first login, you may be asked to complete a short diagnostic test. This helps the app tailor your initial learning path.
3.  **Dashboard**: Your dashboard is your home base. Here you'll see your current progress, recommended next steps, and any announcements from your teacher.

#### Core Activities
*   **Learning with the Knowledge Graph**:
    *   Navigate to the "Knowledge Graph" section.
    *   Tap on a concept node to view its details, including micro-lessons (videos or articles) and related concepts.
    *   Use the graph to visually understand how different topics connect. Follow the recommended path or explore on your own.
*   **Asking the AI Tutor**:
    *   Whenever you have a question, tap the "AI Tutor" button.
    *   Type or speak your question. The AI will provide a detailed explanation, often with examples and references to your textbook.
    *   If you're stuck on a calculation, the AI will guide you step-by-step without giving away the final answer.
*   **Running Virtual Experiments**:
    *   Go to the "Virtual Lab".
    *   Select an experiment, such as "Punnett Square Simulation".
    *   Set the parameters (e.g., parent genotypes) and run the simulation.
    *   Analyze the results and compare them with theoretical predictions.
*   **Solving Case Studies**:
    *   In the "Case Studies" section, choose a real-world problem.
    *   Read the scenario and use the provided data to propose a solution.
    *   Receive instant feedback on your analysis.
*   **Taking Quizzes**:
    *   After each micro-lesson, take a short quiz to check your understanding.
    *   The "Assessment Center" offers longer, adaptive quizzes that adjust to your skill level.
    *   Review your results to see a diagnosis of your strengths and weaknesses.

### 2.2 For Teachers

The app provides a powerful set of tools to enhance your teaching and manage your courses. You can access these features through the app or the web-based administrative backend.

#### Course Management
*   **Setup**: Import your student roster and course structure directly from the Chaoxing platform.
*   **Content Curation**: Upload your own teaching materials (lecture slides, notes, videos) and link them to specific nodes on the knowledge graph.
*   **AI Tutor Training**: Enhance the AI Tutor's knowledge base by providing your own FAQs or specialized documents.

#### In the Classroom
*   **Attendance**: Use the app to take attendance with a single tap (using QR codes, location verification, or a simple check-in).
*   **Live Polls & Quizzes**: Launch quick questions during your lecture to gauge student understanding in real-time. Results are displayed instantly on your screen.

#### Assessments & Analytics
*   **Assignment Creation**: Build and assign homework using the app's question bank or by creating your own questions.
*   **Automated Grading**: Objective questions are graded automatically. For subjective questions, an AI assistant can provide pre-grading suggestions.
*   **Student Progress Tracking**: The "Analytics Dashboard" gives you a bird's-eye view of your class's performance.
    *   Identify which concepts are most challenging for your students.
    *   View detailed "Student Portraits" to understand the learning journey of each individual.
    *   Track progress towards OBE-defined learning outcomes.

---

## 3. Developer Guide

This section is for developers who want to build, run, and contribute to the project.

### 3.1 Architecture

The project follows the **MVVM + Clean Architecture** pattern, promoting a separation of concerns, testability, and maintainability. It is structured as a multi-module project:

*   **/app**: The main application module that integrates all other modules. Handles top-level concerns like navigation and the application lifecycle.
*   **/core**: Contains shared utilities, base classes, and cross-cutting concerns like security, localization, and the event bus.
*   **/domain**: A pure Kotlin module that contains the core business logic, models (entities), and repository interfaces. It has no dependencies on the Android framework.
*   **/data**: Implements the repository interfaces defined in the domain layer. It handles all data operations, fetching from remote sources (Retrofit) and local sources (Room database).
*   **/feature/**: Each feature of the app (e.g., `auth`, `knowledge-graph`, `ai-tutor`) is a self-contained module. This improves build times and encapsulation.

**Key Technologies**:
*   **Language**: Kotlin
*   **Architecture**: MVVM + Clean Architecture
*   **UI**: Android Views (XML) with Material Design. (Ready for migration to Jetpack Compose).
*   **Dependency Injection**: Hilt
*   **Asynchronous Programming**: Coroutines + Flow
*   **Networking**: Retrofit + OkHttp
*   **Database**: Room
*   **CI/CD**: GitHub Actions

### 3.2 Setup and Build

1.  **Prerequisites**:
    *   Android Studio Iguana | 2023.2.1 or newer
    *   JDK 17
    *   (Optional) Docker Desktop for running the backend environment.

2.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd animal-genetics-app
    ```

3.  **Firebase Setup**:
    *   This project requires a Firebase project for Authentication and Analytics.
    *   Create a project on the [Firebase Console](https://console.firebase.google.com/).
    *   Add an Android app with the package name `com.animalgenetics.education`.
    *   Download the `google-services.json` file and place it in the `app/` directory.

4.  **Build the App**:
    *   Open the project in Android Studio.
    *   Let Gradle sync and download all dependencies.
    *   Run the `app` configuration on an emulator or a physical device.

### 3.3 Running the Backend (Local Development)

The backend services can be run locally using Docker Compose.

1.  **Prerequisites**: Docker Desktop installed and running.
2.  **Environment Variables**: The backend requires a JWT secret. Create a `.env` file in the project root:
    ```
    JWT_SECRET=your-super-secret-key-for-local-dev
    ```
3.  **Start the services**:
    ```bash
    docker-compose up --build
    ```
    This will start:
    *   A Postgres database on port `5432`.
    *   A Redis cache on port `6379`.
    *   A MinIO object storage server on port `9000` (API) and `9001` (Console).
    *   The custom API backend on port `8080`.

---

## 4. Administration Guide

This section is for system administrators responsible for deploying and maintaining the production environment.

### 4.1 Backend Deployment

The backend is designed to be deployed as a set of containerized services, suitable for cloud platforms like AWS, Google Cloud, or Azure using Kubernetes or a similar orchestrator.

*   **Database**: A managed PostgreSQL instance (e.g., AWS RDS) is recommended for production.
*   **Cache**: A managed Redis instance (e.g., AWS ElastiCache) is recommended.
*   **Object Storage**: A service like Amazon S3 or Google Cloud Storage should be used.
*   **API Server**: The backend container can be deployed to a container orchestration service like Amazon EKS, Google GKE, or a PaaS like Heroku.

### 4.2 CI/CD

The repository includes a GitHub Actions workflow (`.github/workflows/android-deploy.yml`) that automates testing and deployment:

*   **On Pull Request**: The workflow runs unit tests and instrumentation tests.
*   **On Merge to `main`**:
    1.  Tests are run again.
    2.  The Android App Bundle (`.aab`) is built and signed using secrets stored in GitHub.
    3.  The signed bundle is uploaded to the Google Play Console's internal testing track.
    4.  A new release is created on GitHub.

**Required Secrets for CI/CD**:
*   `KEYSTORE_BASE64`: The Base64-encoded Java Keystore file.
*   `KEYSTORE_PASSWORD`: The password for the keystore.
*   `KEY_ALIAS`: The alias of the signing key.
*   `KEY_PASSWORD`: The password for the key.
*   `GOOGLE_PLAY_SERVICE_ACCOUNT`: The JSON service account key for authenticating with the Google Play Developer API.
