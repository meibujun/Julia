package com.animalgenetics.data.repository

import com.animalgenetics.domain.model.Achievement
import com.animalgenetics.domain.model.AchievementTrigger
import com.animalgenetics.domain.repository.AchievementRepository
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class AchievementRepositoryImpl @Inject constructor() : AchievementRepository {
    override suspend fun getAchievementsForTrigger(trigger: AchievementTrigger): List<Achievement> {
        // Mock implementation
        return emptyList()
    }

    override suspend fun hasAchievement(userId: String, achievementId: String): Boolean {
        // Mock implementation
        return false
    }

    override suspend fun awardAchievement(userId: String, achievement: Achievement) {
        // Mock implementation
    }
}
