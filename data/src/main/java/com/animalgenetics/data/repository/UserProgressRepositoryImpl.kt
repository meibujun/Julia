package com.animalgenetics.data.repository

import com.animalgenetics.domain.model.GeneticsConcept
import com.animalgenetics.domain.model.InteractionType
import com.animalgenetics.domain.repository.StudentProfile
import com.animalgenetics.domain.repository.UserProgressRepository
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class UserProgressRepositoryImpl @Inject constructor() : UserProgressRepository {
    override suspend fun getStudentProfile(userId: String): StudentProfile {
        // Mock implementation
        return StudentProfile(userId, "1st Year", emptyMap())
    }

    override suspend fun recordInteraction(
        userId: String,
        topic: GeneticsConcept,
        interactionType: InteractionType,
        success: Boolean
    ) {
        // Mock implementation
    }

    override suspend fun addPoints(userId: String, points: Int) {
        // Mock implementation
    }

    override suspend fun getTotalPoints(userId: String): Int {
        // Mock implementation
        return 0
    }

    override suspend fun hasMission(userId: String, missionId: String): Boolean {
        // Mock implementation
        return false
    }
}
