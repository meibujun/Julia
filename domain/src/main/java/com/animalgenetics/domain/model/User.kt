package com.animalgenetics.domain.model

data class User(
    val id: String,
    val chaoxingId: String,
    val email: String,
    val displayName: String,
    val role: UserRole,
    val institution: String,
    val profileData: UserProfileData
)

enum class UserRole {
    STUDENT,
    TEACHER
}

data class UserProfileData(
    val grade: String?,
    val major: String?,
    val studentId: String?
)
