package com.animalgenetics.domain.model

data class Question(
    val id: String,
    val conceptId: String,
    val text: String,
    val options: List<String>,
    val correctAnswer: String,
    val difficulty: Float // 0.0 to 1.0
)

data class AnswerResult(
    val questionId: String,
    val isCorrect: Boolean,
    val timeSpent: Long, // in milliseconds
    val confidence: Float // 0.0 to 1.0, provided by user
)

data class AdaptiveQuiz(
    val id: String,
    val userId: String,
    val topic: GeneticsConcept,
    val questions: MutableList<Question>,
    @Transient var adaptiveEngine: Any?, // To avoid circular dependency, will be handled by DI
    var currentDifficulty: Float,
    var estimatedAbility: Float = 0.5f // User's estimated ability level
)

data class AssessmentFeedback(
    val isCorrect: Boolean,
    val explanation: String,
    val hints: List<String>,
    val relatedConcepts: List<GeneticsConcept>,
    val recommendedReview: List<Any> // Could be links to micro-lessons, etc.
)
