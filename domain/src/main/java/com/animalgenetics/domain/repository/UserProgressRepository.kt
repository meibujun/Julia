package com.animalgenetics.domain.repository

import com.animalgenetics.domain.model.GeneticsConcept
import com.animalgenetics.domain.model.InteractionType

// A placeholder for the student's profile/progress
data class StudentProfile(
    val userId: String,
    val gradeLevel: String,
    val masteryByConcept: Map<String, Float>
) {
    fun getMasteryLevel(topic: GeneticsConcept): Float {
        return masteryByConcept[topic.id] ?: 0.0f
    }
}

interface UserProgressRepository {
    suspend fun getStudentProfile(userId: String): StudentProfile
    suspend fun recordInteraction(userId: String, topic: GeneticsConcept, interactionType: InteractionType, success: Boolean)
    suspend fun addPoints(userId: String, points: Int)
    suspend fun getTotalPoints(userId: String): Int
    suspend fun hasMission(userId: String, missionId: String): Boolean
}
