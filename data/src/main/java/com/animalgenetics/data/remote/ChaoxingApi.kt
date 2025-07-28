package com.animalgenetics.data.remote

import com.animalgenetics.domain.model.ChaoxingAuthRequest
import com.animalgenetics.domain.model.ChaoxingSubmission
import com.animalgenetics.domain.model.ChaoxingToken
import com.animalgenetics.domain.model.LearningAnalytics
import com.animalgenetics.domain.model.SubmissionResult
import retrofit2.http.Body
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.PUT
import retrofit2.http.Path

interface ChaoxingApi {
    @POST("auth")
    suspend fun authenticate(@Body request: ChaoxingAuthRequest): ChaoxingToken

    @POST("submissions")
    suspend fun submitAssignment(@Body submission: ChaoxingSubmission): SubmissionResult

    @PUT("analytics/{userId}")
    suspend fun updateStudentAnalytics(@Path("userId") userId: String, @Body analytics: Map<String, Any>)

    // Placeholder endpoints for other sync operations
    @GET("attendance/{studentId}")
    suspend fun getAttendance(@Path("studentId") studentId: String): List<Any>

    @GET("grades/{studentId}")
    suspend fun getGrades(@Path("studentId") studentId: String): List<Any>

    @GET("progress/{studentId}")
    suspend fun getCourseProgress(@Path("studentId") studentId: String): Any
}
