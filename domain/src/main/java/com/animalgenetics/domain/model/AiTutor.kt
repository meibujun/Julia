package com.animalgenetics.domain.model

data class AiResponse(
    val answer: String,
    val suggestedFollowUp: List<String>,
    val relatedConcepts: List<GeneticsConcept>,
    val confidence: Float
)

data class LearningContext(
    val currentTopic: GeneticsConcept,
    val preferredLanguage: String, // e.g., "en-US", "zh-CN"
    val preferOnDevice: Boolean
)

enum class InteractionType {
    AI_QUERY,
    QUIZ_ATTEMPT,
    VIRTUAL_LAB_SESSION
}
