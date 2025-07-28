package com.animalgenetics.feature.gamification

import com.animalgenetics.core.EventBus
import com.animalgenetics.core.QuizCompletedEvent
import com.animalgenetics.domain.model.Achievement
import com.animalgenetics.domain.model.AchievementTrigger
import com.animalgenetics.domain.model.Requirement
import com.animalgenetics.domain.repository.AchievementRepository
import javax.inject.Inject

class AchievementTracker @Inject constructor(
    private val eventBus: EventBus,
    private val repository: AchievementRepository
) {

    init {
        // In a real app, this subscription would be managed by a lifecycle owner
        // For example, in an Application class or a dedicated service
        eventBus.subscribe<QuizCompletedEvent> { event ->
            // This launch should be tied to a proper coroutine scope
            // GlobalScope.launch { checkAchievements(event.userId, AchievementTrigger.QUIZ_COMPLETED, event) }
        }

        eventBus.subscribe<com.animalgenetics.core.ConceptMasteredEvent> { event ->
            // GlobalScope.launch { checkAchievements(event.userId, AchievementTrigger.CONCEPT_MASTERED, event) }
        }
    }

    private suspend fun checkAchievements(
        userId: String,
        trigger: AchievementTrigger,
        eventData: Any
    ) {
        val relevantAchievements = repository.getAchievementsForTrigger(trigger)

        relevantAchievements.forEach { achievement ->
            if (!repository.hasAchievement(userId, achievement.id)) {
                val qualified = achievement.requirements.all { requirement ->
                    checkRequirement(userId, requirement, eventData)
                }

                if (qualified) {
                    awardAchievement(userId, achievement)
                }
            }
        }
    }

    private fun checkRequirement(userId: String, requirement: Requirement, eventData: Any): Boolean {
        // Placeholder for requirement checking logic
        return true
    }

    private suspend fun awardAchievement(userId: String, achievement: Achievement) {
        repository.awardAchievement(userId, achievement)
        // Post an event to notify the UI
        eventBus.post(AchievementUnlockedEvent(userId, achievement))
    }
}

data class AchievementUnlockedEvent(val userId: String, val achievement: Achievement)
