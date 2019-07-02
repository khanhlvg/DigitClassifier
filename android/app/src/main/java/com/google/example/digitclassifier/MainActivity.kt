package com.google.example.digitclassifier

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.divyanshu.draw.widget.DrawView

class MainActivity : AppCompatActivity() {

    private var drawView: DrawView? = null
    private var clearButton: Button? = null
    private var predictedTextView: TextView? = null
    private var digitClassifier: DigitClassifier? = null

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Setup view instances
        drawView = findViewById(R.id.draw_view)
        drawView?.setStrokeWidth(150.0f)
        clearButton = findViewById(R.id.clear_button)
        predictedTextView = findViewById(R.id.predicted_text)

        // Setup clear drawing button
        clearButton?.setOnClickListener {
            drawView?.clearCanvas()
            predictedTextView?.text = getString(R.string.prediction_text_placeholder)
        }

        // Setup classification trigger so that it classify after every stroke drew
        drawView?.setOnTouchListener { _, event ->
            // As we have interrupted DrawView's touch event,
            // we first need to pass touch events through to the instance for the drawing to show up
            drawView?.onTouchEvent(event)

            // Then if user finished a touch event, run classification
            if (event.action == MotionEvent.ACTION_UP) {
                classifyDrawing()
            }

            true
        }

        // Setup digit classifier
        val classifier = DigitClassifier(this)
        classifier
            .initialize()
            .addOnSuccessListener { digitClassifier = classifier }
            .addOnFailureListener { e -> Log.e(TAG, "Unable to setup digit classifier.", e) }
    }

    override fun onDestroy() {
        digitClassifier?.close()
        super.onDestroy()
    }

    private fun classifyDrawing() {
        val bitmap = drawView?.getBitmap()
        val classifier = digitClassifier

        if ((bitmap != null) && (classifier != null)) {
            classifier
                .classifyAsync(bitmap)
                .addOnSuccessListener { resultText -> predictedTextView?.text = resultText }
                .addOnFailureListener { e -> Log.e(TAG, "Error classifying drawing.", e) }
        }
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
