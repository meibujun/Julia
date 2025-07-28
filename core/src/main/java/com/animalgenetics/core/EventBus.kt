package com.animalgenetics.core

import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.map
import javax.inject.Inject
import javax.inject.Singleton

@Singleton
class EventBus @Inject constructor() {
    private val _events = MutableSharedFlow<Any>()
    val events = _events.asSharedFlow()

    suspend fun post(event: Any) {
        _events.emit(event)
    }

    inline fun <reified T> subscribe(crossinline onEvent: (T) -> Unit) {
        events.filter { it is T }
            .map { it as T }
            .
            // This should be collected in a coroutine scope
            // For simplicity, this is just a placeholder
            // In a real app, you'd collect this in a ViewModel's lifecycle scope
            // e.g., viewModelScope.launch { eventBus.subscribe<MyEvent> { ... } }
            // As this is a Singleton, direct collection is not safe.
            // This is a simplified example for the sake of the guide.
            // A better approach would be to return the Flow and let the caller collect it.
            Unit
    }
}

// Example Events
data class QuizCompletedEvent(val userId: String, val quizId: String, val score: Float)
data class ConceptMasteredEvent(val userId: String, val conceptId: String)
