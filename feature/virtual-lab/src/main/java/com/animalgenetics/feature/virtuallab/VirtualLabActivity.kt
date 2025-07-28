package com.animalgenetics.feature.virtuallab

import android.os.Bundle
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.animalgenetics.domain.model.ExperimentAction
import com.animalgenetics.domain.model.ExperimentConfig
import com.animalgenetics.domain.model.UnityCallable
import com.google.gson.Gson
import dagger.hilt.android.AndroidEntryPoint

// Placeholder for the actual UnityPlayer class
class UnityPlayer(context: android.content.Context) : android.widget.FrameLayout(context) {
    companion object {
        fun UnitySendMessage(gameObject: String, methodName: String, message: String) {
            println("UnitySendMessage: $gameObject, $methodName, $message")
        }
    }
}

@AndroidEntryPoint
class VirtualLabActivity : AppCompatActivity() {

    private lateinit var unityPlayer: UnityPlayer
    private val viewModel: VirtualLabViewModel by viewModels()

    companion object {
        const val EXTRA_EXPERIMENT_TYPE = "EXTRA_EXPERIMENT_TYPE"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize Unity player for 3D simulations
        unityPlayer = UnityPlayer(this)
        setContentView(unityPlayer)

        // Set up experiment based on selected topic
        val experimentType = intent.getStringExtra(EXTRA_EXPERIMENT_TYPE)
        viewModel.loadExperiment(experimentType)

        observeExperimentState()
    }

    private fun observeExperimentState() {
        viewModel.experimentState.observe(this) { state ->
            when (state) {
                is com.animalgenetics.domain.model.ExperimentState.Ready -> startExperiment(state.config)
                is com.animalgenetics.domain.model.ExperimentState.InProgress -> updateExperimentUI(state.progress)
                is com.animalgenetics.domain.model.ExperimentState.Completed -> showResults(state.results)
                is com.animalgenetics.domain.model.ExperimentState.Error -> showError(state.message)
            }
        }
    }

    private fun startExperiment(config: ExperimentConfig) {
        // Send configuration to Unity
        UnityPlayer.UnitySendMessage(
            "ExperimentController",
            "LoadExperiment",
            Gson().toJson(config)
        )
    }

    private fun updateExperimentUI(progress: Float) {
        // Could send progress updates to a UI overlay
    }

    private fun showResults(results: com.animalgenetics.domain.model.SimulationResult) {
        // Display results in a dialog or separate screen
    }

    private fun showError(message: String) {
        // Show an error message
    }

    // This method is intended to be called from Unity
    @UnityCallable
    fun onExperimentAction(action: String, data: String) {
        viewModel.processExperimentAction(
            ExperimentAction.fromJson(action, data)
        )
    }
}
