package com.animalgenetics.domain.repository

import com.animalgenetics.domain.model.Achievement
import com.animalgenetics.domain.model.AchievementTrigger

interface AchievementRepository {
    suspend fun getAchievementsForTrigger(trigger: AchievementTrigger): List<Achievement>
    suspend fun hasAchievement(userId: String, achievementId: String): Boolean
    suspend fun awardAchievement(userId: String, achievement: Achievement)
}
