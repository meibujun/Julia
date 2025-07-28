package com.animalgenetics.domain.model

enum class Language {
    CHINESE_SIMPLIFIED,
    CHINESE_TRADITIONAL,
    ENGLISH
}

data class TermTranslation(
    val en: String,
    val zh_CN: String,
    val zh_TW: String
)

data class ChaoxingAuthRequest(
    val username: String,
    val encryptedPass: String
)

data class ChaoxingToken(
    val firebaseToken: String,
    val userId: String,
    val userType: String, // "student" or "teacher"
    val email: String,
    val displayName: String,
    val institution: String,
    val grade: String?,
    val major: String?,
    val studentId: String?
)

data class Assignment(
    val id: String,
    val courseId: String,
    val content: String
)

data class ChaoxingSubmission(
    val studentId: String,
    val courseId: String,
    val assignmentId: String,
    val data: String, // Encrypted data
    val timestamp: Long
)

data class SubmissionResult(
    val success: Boolean,
    val message: String
)

data class LearningAnalytics(
    val userId: String,
    val timeSpent: Long,
    val conceptsMastered: Int
) {
    fun toChaoxingFormat(): Map<String, Any> {
        // Convert to the format expected by Chaoxing API
        return mapOf(
            "student_id" to userId,
            "learning_duration_minutes" to timeSpent / 60000,
            "mastered_concepts_count" to conceptsMastered
        )
    }
}
