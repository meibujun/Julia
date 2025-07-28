package com.animalgenetics.domain.model

data class Achievement(
    val id: String,
    val name: String,
    val description: String,
    val iconUrl: String,
    val category: AchievementCategory,
    val rarity: Rarity,
    val requirements: List<Requirement>
)

enum class AchievementCategory {
    LEARNING,
    EXPLORATION,
    SOCIAL,
    MASTERY
}

enum class Rarity {
    COMMON,
    UNCOMMON,
    RARE,
    EPIC,
    LEGENDARY
}

data class Requirement(
    val trigger: AchievementTrigger,
    val condition: String // e.g., "score > 90", "streak == 7"
)

enum class AchievementTrigger {
    QUIZ_COMPLETED,
    CONCEPT_MASTERED,
    LOGIN,
    SOCIAL_SHARE
}

data class Mission(
    val id: String,
    val title: String
)

data class PointsAwarded(
    val points: Int,
    val multiplier: Float,
    val newLevel: Int,
    val unlockedFeatures: List<String>
)

enum class UserAction {
    COMPLETE_QUIZ,
    MASTER_CONCEPT,
    DAILY_LOGIN
}

data class ActionContext(
    val details: Map<String, Any>
)

sealed class CreativeContribution {
    data class CustomExperiment(val experiment: Any) : CreativeContribution()
    data class StudyNote(val note: Any) : CreativeContribution()
}

enum class LeaderboardType {
    WEEKLY_POINTS,
    TOPIC_MASTERY,
    HELPING_OTHERS,
    STREAK_DAYS
}
