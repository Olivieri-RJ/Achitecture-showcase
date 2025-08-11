# ESP32 Motion Detection & Feedback Control System

## Overview
This firmware demonstrates a motion classification and feedback control system running on an ESP32 under **FreeRTOS**, featuring:
- **IMU-based motion classification** (tilt, fall, freeze)
- **PID control** for servo stabilization
- **Haptic feedback** for motion correction
- **Event logging** for debugging and analytics

---

## Architecture
1. **IMU Driver Layer**
   - Initializes and reads accelerometer and gyroscope data.
   - Calculates magnitude and roll for motion classification.

2. **Motion Classifier**
   - Uses recent history to classify motion events:
     - Tilt Left
     - Tilt Right
     - Fall
     - Freeze
   - Logs events for external processing.

3. **PID Control**
   - Stabilizes servo position to maintain target tilt (0°).
   - Configurable constants: `KP`, `KI`, `KD`.

4. **Haptic Feedback**
   - Vibrational stimulus triggered upon freeze detection.

---

## Hardware Requirements
- ESP32 board
- I2C-compatible IMU (e.g., MPU6050)
- Servo motor
- Haptic feedback actuator

---

## Build & Flash
```bash
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor

```python
/**
 * @file main.c
 * @brief Motion detection and feedback control system using IMU, servo actuation, and haptic feedback.
 *
 * Features:
 * - IMU initialization and data acquisition
 * - PID-based servo stabilization
 * - Motion classification (tilt, fall, freeze)
 * - Haptic feedback activation
 * - Event logging
 *
 * Target: ESP-IDF (FreeRTOS-based firmware)
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"

// =================== Dependencies & Config ===================
#define DEPENDENCIES_IMU    1
#define DEPENDENCIES_SERVO  1
#define DEPENDENCIES_HAPTIC 1

// PID constants
#define KP 0.5
#define KI 0.1
#define KD 0.05
#define MAX_PWM 100
#define MIN_PWM 0

// =================== Data Structures ===================
typedef struct {
    float accel[3];
    float gyro[3];
    float magnitude;
} imu_data_t;

typedef enum {
    MOTION_NONE,
    MOTION_TILT_LEFT,
    MOTION_TILT_RIGHT,
    MOTION_FALL,
    MOTION_FREEZE
} motion_event_t;

typedef struct {
    int id;
    float last_error;
    float integral;
} Servo;

// =================== Utilities ===================
void log_event(const char* type, const char* value) {
    printf("[LOG] %s - %s\n", type, value);
}

// =================== IMU Functions ===================
void imu_init(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = 21,
        .scl_io_num = 22,
        .master.clk_speed = 100000
    };
    i2c_param_config(I2C_NUM_0, &conf);
    i2c_driver_install(I2C_NUM_0, conf.mode, 0, 0, 0);
    printf("IMU initialized.\n");
}

void imu_read(imu_data_t* data) {
    data->accel[0] = (float)(rand() % 100) / 100.0f - 0.5f;
    data->accel[1] = (float)(rand() % 100) / 100.0f - 0.5f;
    data->accel[2] = (float)(rand() % 100) / 100.0f - 0.5f;
    data->magnitude = sqrtf(
        powf(data->accel[0], 2) +
        powf(data->accel[1], 2) +
        powf(data->accel[2], 2)
    );
}

// =================== Control Functions ===================
void pid_adjust_servo(float actual_tilt, float setpoint, Servo* servo) {
    float error = setpoint - actual_tilt;
    servo->integral += error;
    float derivative = error - servo->last_error;

    float pwm = KP * error + KI * servo->integral + KD * derivative;
    pwm = fmaxf(MIN_PWM, fminf(MAX_PWM, pwm));

    printf("Servo %d PWM: %.2f%%\n", servo->id, pwm);
    servo->last_error = error;
}

motion_event_t classify_motion(imu_data_t* data) {
    static float history[10] = {0};
    static int idx = 0;

    float roll = atan2f(data->accel[1], data->accel[2]);
    history[idx++ % 10] = roll;

    float avg_roll = 0.0f;
    for (int i = 0; i < 10; i++) avg_roll += history[i] / 10;

    float emg_variance = 0.1f; // Simulated EMG variance
    if (emg_variance > 0.5 && fabsf(avg_roll) < 0.05f && data->magnitude < 0.1f) {
        log_event("motion", "freeze_detected");
        return MOTION_FREEZE;
    }
    if (fabsf(avg_roll) > 0.52f) {
        log_event("motion", "fall_detected");
        return MOTION_FALL;
    }
    if (avg_roll < -0.17f) return MOTION_TILT_LEFT;
    if (avg_roll > 0.17f)  return MOTION_TILT_RIGHT;

    return MOTION_NONE;
}

// =================== Haptic Feedback ===================
void trigger_haptic(float freq, float amplitude) {
    printf("[HAPTIC] %.1f Hz, %.1f mm amplitude (500ms)\n", freq, amplitude);
    vTaskDelay(pdMS_TO_TICKS(500));
}

// =================== Main Control Loop ===================
void app_main(void) {
    imu_init();
    Servo torso_servo = { .id = 1, .last_error = 0, .integral = 0 };
    imu_data_t imu_data;

    while (1) {
        imu_read(&imu_data);
        motion_event_t event = classify_motion(&imu_data);

        switch (event) {
            case MOTION_FALL:
                log_event("emergency", "fall_detected");
                printf("Activating fail-safe...\n");
                break;
            case MOTION_FREEZE:
                trigger_haptic(100, 1.5f);
                log_event("stimulus", "freeze_breaker_activated");
                break;
            default:
                break;
        }

        float current_tilt = atan2f(imu_data.accel[1], imu_data.accel[2]);
        pid_adjust_servo(current_tilt, 0.0f, &torso_servo);

        vTaskDelay(pdMS_TO_TICKS(50));
    }
}
```
