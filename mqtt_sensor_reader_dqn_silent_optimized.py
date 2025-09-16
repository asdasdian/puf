#!/usr/bin/env python3
"""
MQTT Sensor Data Reader with PUF Obfuscation - DQN Priority System (Silent Mode - Optimized)
Date: 2025-09-14
Author: foladlo
"""

import paho.mqtt.client as mqtt
import json
import time
import sys
import threading
import socket
import psutil
import signal
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
from datetime import datetime, timedelta
from puf_interface import PUFInterface, create_puf_interface, get_available_serial_ports

class DQN(nn.Module):
    def __init__(self, state_size=12, action_size=2, hidden_size=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size=12, action_size=2, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)  # Reduced from 10000
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Faster decay from 0.995
        self.learning_rate = learning_rate
        self.batch_size = 16  # Reduced from 32
        self.target_update = 100
        self.training_steps = 0
        
        # Networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Copy weights to target network
        self.update_target_network()
        
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state, sensor_priority, location_priority):
        # Emergency override for deadline approaching
        time_ratio = state[0]
        urgency = state[1]
        is_critical = state[10]
        
        # Force transmit if very urgent or critical
        if time_ratio < 0.1 or (is_critical > 0.5 and time_ratio < 0.5):
            return 1
        
        # Early training phase - priority-based decisions
        if self.training_steps < 200:
            battery_priority = state[11]
            
            base_prob = 0.6
            priority_boost = (sensor_priority * location_priority) / 12.0
            urgency_boost = urgency * 0.3
            critical_boost = is_critical * 0.4
            battery_boost = battery_priority * 0.2
            
            transmission_prob = min(0.95, base_prob + priority_boost + urgency_boost + critical_boost + battery_boost)
            
            if random.random() < transmission_prob:
                return 1
            else:
                return 0
        
        # Override for critical sensors or critical values or low battery
        if sensor_priority >= 3.0 or state[10] > 0.5 or state[11] > 0.8:
            if urgency > 0.8 and random.random() < 0.9:
                return 1
        
        # Adjust epsilon based on priority, critical status, and battery
        priority_factor = sensor_priority * location_priority / 3.0
        critical_factor = 1.0 + state[10]
        battery_factor = 1.0 + state[11]
        
        adjusted_epsilon = self.epsilon / (priority_factor * critical_factor * battery_factor)
        
        if random.random() <= adjusted_epsilon:
            return random.randrange(self.action_size)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        
        # Optimized tensor conversion to avoid warning
        states_array = np.array([e[0] for e in batch])
        actions_array = np.array([e[1] for e in batch])
        rewards_array = np.array([e[2] for e in batch])
        next_states_array = np.array([e[3] for e in batch])
        dones_array = np.array([e[4] for e in batch])
        
        states = torch.FloatTensor(states_array).to(self.device)
        actions = torch.LongTensor(actions_array).to(self.device)
        rewards = torch.FloatTensor(rewards_array).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        dones = torch.BoolTensor(dones_array).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (next_q_values * 0.99 * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.training_steps += 1
        
        if self.training_steps % self.target_update == 0:
            self.update_target_network()
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MQTTSensorReader:
    def __init__(self, puf_interface):
        self.puf = puf_interface
        try:
            self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION1)
        except:
            self.client = mqtt.Client()
        
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.log_file = f"sensor_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.performance_file = f"performance_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # DQN Agent
        self.dqn_agent = DQNAgent(state_size=12)
        self.max_delay = 300  # Maximum deadline (blood_pressure)
        
        # Priority levels
        self.sensor_priority = {
            "heart_rate": 3.0,
            "spo2": 3.0,
            "blood_pressure": 3.0,
            "body_temperature": 2.0,
            "respiratory_rate": 2.0,
            "ambient_temperature": 1.0,
            "humidity": 1.0
        }
        
        self.location_priority = {
            "operating_room": 4.0,
            "icu": 3.0,
            "emergency": 2.5,
            "ward": 1.0
        }
        
        # Critical value ranges
        self.critical_ranges = {
            "heart_rate": {"critical_low": 50, "critical_high": 120},
            "spo2": {"critical_low": 90, "critical_high": 100},
            "blood_pressure": {"critical_systolic_high": 160, "critical_diastolic_high": 100},
            "body_temperature": {"critical_low": 35.0, "critical_high": 39.0},
            "respiratory_rate": {"critical_low": 8, "critical_high": 25},
            "ambient_temperature": {"critical_low": 18.0, "critical_high": 28.0},
            "humidity": {"critical_low": 30, "critical_high": 70}
        }
        
        # Sensor deadlines (in seconds)
        self.sensor_deadlines = {
            "heart_rate": 1,
            "spo2": 1,
            "blood_pressure": 300,
            "body_temperature": 10,
            "respiratory_rate": 10,
            "ambient_temperature": 20,
            "humidity": 20
        }
        
        # Performance monitoring variables
        self.messages_processed = 0
        self.messages_failed = 0
        self.start_time = time.time()
        self.last_performance_check = time.time()
        self.max_processing_rate = 0
        self.performance_lock = threading.Lock()
        
        # Statistics tracking
        self.sensors_read_count = 0
        self.sensors_processed_count = 0
        self.critical_sensors_count = 0
        
        # Delay tracking
        self.delay_stats = {
            'all': [],
            'by_sensor_type': defaultdict(list),
            'by_location': defaultdict(list),
            'critical_values': []
        }
        
        # Fixed deadline tracking
        self.sensor_last_received = {}
        self.sensor_deadlines_missed = {}
        self.sensor_total_reads = {}
        self.sensor_successful_transfers = {}
        self.missed_sensor_tracking = {}  # Track which sensors already counted as missed
        self.missed_sensor_values = []
        
        # Shutdown flag
        self.shutdown_flag = threading.Event()
        
        # Initialize counters for each sensor type
        for sensor_type in self.sensor_deadlines.keys():
            self.sensor_deadlines_missed[sensor_type] = 0
            self.sensor_total_reads[sensor_type] = 0
            self.sensor_successful_transfers[sensor_type] = 0
        
        # Start background threads with reduced frequency
        self.performance_thread = threading.Thread(target=self._monitor_performance, daemon=True)
        self.performance_thread.start()
        
        self.deadline_thread = threading.Thread(target=self._monitor_deadlines, daemon=True)
        self.deadline_thread.start()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.log_to_file("Shutdown signal received")
        self.shutdown_gracefully()
        
    def shutdown_gracefully(self):
        """Perform graceful shutdown"""
        self.shutdown_flag.set()
        
        # Save DQN model
        torch.save(self.dqn_agent.q_network.state_dict(), f"dqn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        
        # Log final statistics
        self.log_deadline_statistics()
        
        max_rate = self.get_max_processing_rate()
        final_message = f"Final Max Processing Rate: {max_rate:.2f} msg/sec"
        self.log_to_file(final_message)
        
        # Stop MQTT client
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        
        # Disconnect PUF
        if self.puf:
            self.puf.disconnect()
            
        self.log_to_file("Graceful shutdown completed")
        sys.exit(0)
    
    def is_critical_value(self, sensor_type, value):
        """Check if sensor value is in critical range"""
        try:
            if sensor_type not in self.critical_ranges:
                return False
            
            ranges = self.critical_ranges[sensor_type]
            
            if sensor_type == "blood_pressure" and isinstance(value, dict):
                systolic = value.get('systolic', 0)
                diastolic = value.get('diastolic', 0)
                return (systolic > ranges.get('critical_systolic_high', 999) or 
                        diastolic > ranges.get('critical_diastolic_high', 999))
            
            elif isinstance(value, (int, float)):
                low = ranges.get('critical_low', 0)
                high = ranges.get('critical_high', 999)
                return value < low or value > high
            
            return False
            
        except Exception:
            return False
    
    def get_battery_priority(self, battery_level):
        """Get battery priority based on level"""
        if battery_level < 15:
            return 3.0  # High priority for low battery
        elif battery_level < 50:
            return 2.0  # Medium priority
        else:
            return 1.0  # Low priority for good battery
    
    def get_state(self, sensor_type, location, current_time, time_since_last, remaining_time, sensor_value=None, battery_level=100):
        """Generate state vector for DQN"""
        try:
            # Basic features
            time_ratio = remaining_time / self.max_delay
            urgency = 1.0 - time_ratio
            priority_norm = self.sensor_priority.get(sensor_type, 1.0) / 3.0
            location_norm = self.location_priority.get(location, 1.0) / 4.0
            time_of_day = (current_time % 86400) / 86400
            time_since_norm = min(time_since_last / self.max_delay, 1.0)
            training_progress = min(float(self.dqn_agent.training_steps) / 1000.0, 1.0)
            noise = random.random()
            
            # Circular time features
            sin_time = np.sin(2 * np.pi * time_of_day)
            cos_time = np.cos(2 * np.pi * time_of_day)
            
            # Critical value feature
            is_critical = 1.0 if self.is_critical_value(sensor_type, sensor_value) else 0.0
            
            # Battery priority feature (normalized)
            battery_priority = self.get_battery_priority(battery_level) / 3.0
            
            # State vector (12 dimensions)
            state = np.array([
                time_ratio,           # 0: نسبت زمان باقی‌مانده
                urgency,              # 1: سطح اضطرار
                priority_norm,        # 2: اولویت سنسور
                location_norm,        # 3: اولویت مکان
                time_of_day,          # 4: زمان روز
                time_since_norm,      # 5: زمان از آخرین ارسال
                training_progress,    # 6: پیشرفت آموزش
                noise,                # 7: نویز تصادفی
                sin_time,             # 8: ویژگی چرخه‌ای
                cos_time,             # 9: ویژگی چرخه‌ای
                is_critical,          # 10: وضعیت بحرانی
                battery_priority      # 11: اولویت باتری
            ])
            
            return state
            
        except Exception as e:
            self.log_to_file(f"Error generating state: {e}")
            return np.zeros(12)
    
    def calculate_reward(self, action, sensor_type, location, time_remaining, sensor_value, battery_level):
        """Calculate reward for DQN training"""
        try:
            base_priority = self.sensor_priority.get(sensor_type, 1.0) * self.location_priority.get(location, 1.0)
            time_ratio = max(time_remaining / self.max_delay, 0.0)
            urgency_factor = 1.0 + (1.0 - time_ratio) * 2.0
            
            # Critical value multiplier
            critical_multiplier = 2.0 if self.is_critical_value(sensor_type, sensor_value) else 1.0
            
            # Battery multiplier
            battery_multiplier = self.get_battery_priority(battery_level)
            
            if action == 1:  # Transmitted
                if time_remaining > 0:
                    reward = base_priority * urgency_factor * critical_multiplier * battery_multiplier
                    
                    # Bonus for critical sensors transmitted on time
                    if self.sensor_priority.get(sensor_type, 1.0) >= 3.0 and time_ratio > 0.5:
                        reward *= 1.5
                    
                    # Extra bonus for critical values
                    if self.is_critical_value(sensor_type, sensor_value):
                        reward *= 1.3
                    
                    # Extra bonus for low battery sensors
                    if battery_level < 15:
                        reward *= 1.4
                        
                else:
                    # Late transmission penalty
                    reward = base_priority * critical_multiplier * battery_multiplier * 0.3
                    
            else:  # Not transmitted
                if time_remaining > 0:
                    penalty = -0.2 * base_priority * urgency_factor * critical_multiplier * battery_multiplier
                else:
                    # Severe penalty for missing critical data
                    penalty = -2.0 * base_priority * critical_multiplier * battery_multiplier
                    
                    # Extra severe penalty for critical values
                    if self.is_critical_value(sensor_type, sensor_value):
                        penalty *= 1.5
                    
                    # Extra penalty for low battery sensors missing deadline
                    if battery_level < 15:
                        penalty *= 1.3
                        
                reward = penalty
            
            return reward
            
        except Exception as e:
            self.log_to_file(f"Error calculating reward: {e}")
            return 0.0
    
    def calculate_delay_stats(self, delay_list):
        """Calculate delay statistics (min, max, avg)"""
        if not delay_list:
            return {"min": 0.0, "max": 0.0, "avg": 0.0}
        
        return {
            "min": min(delay_list),
            "max": max(delay_list),
            "avg": sum(delay_list) / len(delay_list)
        }
    
    def log_to_file(self, message):
        """Log message to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def log_performance(self, ram_mb, processing_rate, packet_loss_rate):
        """Log performance metrics to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.performance_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] RAM: {ram_mb:.2f} MB, Processing Rate: {processing_rate:.2f} msg/sec, Packet Loss: {packet_loss_rate:.2f}%, Sensors Read: {self.sensors_read_count}, Sensors Processed: {self.sensors_processed_count}, Critical Sensors: {self.critical_sensors_count}\n")
    
    def log_dqn_neural_network_parameters(self):
        """Log DQN neural network parameters and state"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.performance_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}] DQN NEURAL NETWORK PARAMETERS:\n")
            f.write("="*80 + "\n")
            
            # Network architecture
            f.write(f"Network Architecture:\n")
            f.write(f"- State Size: {self.dqn_agent.state_size}\n")
            f.write(f"- Action Size: {self.dqn_agent.action_size}\n")
            f.write(f"- Hidden Size: 64\n")
            f.write(f"- Device: {self.dqn_agent.device}\n")
            
            # Training parameters
            f.write(f"\nTraining Parameters:\n")
            f.write(f"- Learning Rate: {self.dqn_agent.learning_rate}\n")
            f.write(f"- Batch Size: {self.dqn_agent.batch_size}\n")
            f.write(f"- Memory Size: {len(self.dqn_agent.memory)}/{self.dqn_agent.memory.maxlen}\n")
            f.write(f"- Training Steps: {self.dqn_agent.training_steps}\n")
            f.write(f"- Target Update Frequency: {self.dqn_agent.target_update}\n")
            
            # Exploration parameters
            f.write(f"\nExploration Parameters:\n")
            f.write(f"- Current Epsilon: {self.dqn_agent.epsilon:.6f}\n")
            f.write(f"- Epsilon Min: {self.dqn_agent.epsilon_min}\n")
            f.write(f"- Epsilon Decay: {self.dqn_agent.epsilon_decay}\n")
            
            # Network weights statistics
            f.write(f"\nMain Network Weights Statistics:\n")
            for name, param in self.dqn_agent.q_network.named_parameters():
                if param.requires_grad:
                    weight_mean = param.data.mean().item()
                    weight_std = param.data.std().item()
                    weight_min = param.data.min().item()
                    weight_max = param.data.max().item()
                    f.write(f"- {name}: Mean={weight_mean:.6f}, Std={weight_std:.6f}, Min={weight_min:.6f}, Max={weight_max:.6f}\n")
            
            # Gradients statistics (if available)
            f.write(f"\nGradient Statistics:\n")
            for name, param in self.dqn_agent.q_network.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_mean = param.grad.mean().item()
                    grad_std = param.grad.std().item()
                    grad_norm = param.grad.norm().item()
                    f.write(f"- {name} gradient: Mean={grad_mean:.6f}, Std={grad_std:.6f}, Norm={grad_norm:.6f}\n")
                else:
                    f.write(f"- {name} gradient: No gradient available\n")
            
            f.write("="*80 + "\n\n")
    
    def log_deadline_statistics(self):
        """Log detailed deadline statistics to file"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.performance_file, 'a', encoding='utf-8') as f:
            f.write(f"\n[{timestamp}] FINAL STATISTICS:\n")
            f.write("="*80 + "\n")
            
            total_successful = sum(self.sensor_successful_transfers.values())
            total_reads = sum(self.sensor_total_reads.values())
            total_missed = sum(self.sensor_deadlines_missed.values())
            
            f.write(f"تعداد کل سنسورهای خوانده شده: {self.sensors_read_count}\n")
            f.write(f"تعداد کل سنسورهای پردازش شده: {self.sensors_processed_count}\n")
            f.write(f"تعداد سنسورهای با مقدار بحرانی: {self.critical_sensors_count}\n")
            f.write(f"تعداد انتقال موفق: {total_successful}\n")
            f.write(f"کل خوانش‌ها: {total_reads}\n")
            f.write(f"تعداد deadline های از دست رفته: {total_missed}\n")
            f.write(f"DQN Training Steps: {self.dqn_agent.training_steps}\n")
            f.write(f"Current Epsilon: {self.dqn_agent.epsilon:.4f}\n")
            
            # Log neural network parameters
            self.log_dqn_neural_network_parameters()
            
            # Delay statistics
            f.write("\nآمار تاخیر (ثانیه):\n")
            f.write("-" * 40 + "\n")
            
            # Overall delay stats
            overall_stats = self.calculate_delay_stats(self.delay_stats['all'])
            f.write(f"کلی - میانگین: {overall_stats['avg']:.2f}s, کمینه: {overall_stats['min']:.2f}s, بیشینه: {overall_stats['max']:.2f}s\n")
            
            # Delay by sensor type
            f.write("\nبر اساس نوع سنسور:\n")
            for sensor_type, delays in self.delay_stats['by_sensor_type'].items():
                stats = self.calculate_delay_stats(delays)
                f.write(f"{sensor_type} - میانگین: {stats['avg']:.2f}s, کمینه: {stats['min']:.2f}s, بیشینه: {stats['max']:.2f}s, تعداد: {len(delays)}\n")
            
            # Delay by location
            f.write("\nبر اساس مکان:\n")
            for location, delays in self.delay_stats['by_location'].items():
                stats = self.calculate_delay_stats(delays)
                f.write(f"{location} - میانگین: {stats['avg']:.2f}s, کمینه: {stats['min']:.2f}s, بیشینه: {stats['max']:.2f}s, تعداد: {len(delays)}\n")
            
            # Critical values delay
            critical_stats = self.calculate_delay_stats(self.delay_stats['critical_values'])
            f.write(f"\nسنسورهای با مقدار بحرانی - میانگین: {critical_stats['avg']:.2f}s, کمینه: {critical_stats['min']:.2f}s, بیشینه: {critical_stats['max']:.2f}s, تعداد: {len(self.delay_stats['critical_values'])}\n")
            
            f.write("\nجزئیات هر نوع سنسور:\n")
            f.write("-" * 40 + "\n")
            
            for sensor_type in self.sensor_deadlines.keys():
                deadline = self.sensor_deadlines[sensor_type]
                successful = self.sensor_successful_transfers[sensor_type]
                total = self.sensor_total_reads[sensor_type]
                missed = self.sensor_deadlines_missed[sensor_type]
                
                f.write(f"{sensor_type}:\n")
                f.write(f"  Deadline: {deadline} ثانیه\n")
                f.write(f"  انتقال موفق: {successful}\n")
                f.write(f"  کل خوانش: {total}\n")
                f.write(f"  از دست رفته: {missed}\n")
                
                # Fixed success rate calculation
                if total > 0:
                    success_rate = (successful / total) * 100
                    f.write(f"  نرخ موفقیت: {success_rate:.2f}%\n")
                else:
                    f.write(f"  نرخ موفقیت: 0.00%\n")
                f.write("\n")
            
            # Log missed sensor values
            if self.missed_sensor_values:
                f.write("مقادیر سنسورهای از دست رفته:\n")
                f.write("-" * 40 + "\n")
                for missed_value in self.missed_sensor_values:
                    f.write(f"[{missed_value['timestamp']}] {missed_value['sensor_id']} ({missed_value['sensor_type']}) - Expected deadline: {missed_value['deadline']}s - Time since last: {missed_value['time_since_last']:.2f}s\n")
            else:
                f.write("مقادیر سنسورهای از دست رفته:\n")
                f.write("-" * 40 + "\n")
                f.write("هیچ سنسوری از deadline خود عقب نیفتاده است\n")
            
            f.write("="*80 + "\n\n")
    
    def _monitor_deadlines(self):
        """Monitor sensor deadlines and detect missed readings - FIXED VERSION"""
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(10)  # Reduced frequency from 5 to 10 seconds
                current_time = datetime.now()
                
                with self.performance_lock:
                    for sensor_id, last_received in list(self.sensor_last_received.items()):
                        sensor_type = self._get_sensor_type_from_id(sensor_id)
                        
                        if sensor_type in self.sensor_deadlines:
                            deadline_seconds = self.sensor_deadlines[sensor_type]
                            time_since_last = (current_time - last_received).total_seconds()
                            
                            # Only count as missed once per sensor per deadline period
                            if time_since_last > deadline_seconds:
                                if sensor_id not in self.missed_sensor_tracking:
                                    self.sensor_deadlines_missed[sensor_type] += 1
                                    
                                    missed_info = {
                                        'sensor_id': sensor_id,
                                        'sensor_type': sensor_type,
                                        'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                                        'deadline': deadline_seconds,
                                        'time_since_last': time_since_last
                                    }
                                    self.missed_sensor_values.append(missed_info)
                                    
                                    # Mark as missed to prevent double counting
                                    self.missed_sensor_tracking[sensor_id] = current_time
                            else:
                                # Reset if sensor is back within deadline
                                if sensor_id in self.missed_sensor_tracking:
                                    del self.missed_sensor_tracking[sensor_id]
                                
            except Exception as e:
                pass
    
    def _get_sensor_type_from_id(self, sensor_id):
        """Extract sensor type from sensor ID"""
        sensor_id_lower = sensor_id.lower()
        if "hr_" in sensor_id_lower or "heart" in sensor_id_lower:
            return "heart_rate"
        elif "spo2" in sensor_id_lower:
            return "spo2"
        elif "bp_" in sensor_id_lower or "blood" in sensor_id_lower:
            return "blood_pressure"
        elif "temp_" in sensor_id_lower and "ambient" not in sensor_id_lower:
            return "body_temperature"
        elif "resp_" in sensor_id_lower or "respiratory" in sensor_id_lower:
            return "respiratory_rate"
        elif "ambient" in sensor_id_lower:
            return "ambient_temperature"
        elif "humidity" in sensor_id_lower:
            return "humidity"
        else:
            return "unknown"
    
    def _monitor_performance(self):
        """Monitor performance metrics in background thread"""
        while not self.shutdown_flag.is_set():
            try:
                time.sleep(10)  # Reduced frequency from 5 to 10 seconds
                
                with self.performance_lock:
                    process = psutil.Process()
                    ram_mb = process.memory_info().rss / 1024 / 1024
                    
                    current_time = time.time()
                    time_diff = current_time - self.last_performance_check
                    
                    if time_diff >= 10.0:  # Log every 10 seconds
                        processing_rate = self.messages_processed / time_diff
                        
                        if processing_rate > self.max_processing_rate:
                            self.max_processing_rate = processing_rate
                        
                        total_messages = self.messages_processed + self.messages_failed
                        packet_loss_rate = (self.messages_failed / total_messages * 100) if total_messages > 0 else 0
                        
                        # Console output (only essential info)
                        print(f"Sensors Read: {self.sensors_read_count}, Processed: {self.sensors_processed_count}, Critical: {self.critical_sensors_count}, Rate: {processing_rate:.2f} msg/sec")
                        
                        self.log_performance(ram_mb, processing_rate, packet_loss_rate)
                        
                        self.messages_processed = 0
                        self.messages_failed = 0
                        self.last_performance_check = current_time
                        
            except Exception as e:
                pass
        
    def check_port_available(self, port):
        """Check if port is available"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('localhost', port))
            sock.close()
            return result == 0
        except:
            return False
        
    def on_connect(self, client, userdata, flags, rc):
        message = f"Connected to MQTT broker with result code {rc}"
        self.log_to_file(message)
        client.subscribe("sensors/+/+")
        
    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode()
            
            self.log_to_file(f"Received message on topic: {topic}")
            self.log_to_file(f"Raw payload: {payload}")
            
            try:
                data = json.loads(payload)
                self.handle_sensor_data(topic, data)
                with self.performance_lock:
                    self.messages_processed += 1
                    self.sensors_read_count += 1
            except json.JSONDecodeError:
                self.handle_legacy_format(topic, payload)
                with self.performance_lock:
                    self.messages_processed += 1
                    self.sensors_read_count += 1
                
        except Exception as e:
            message = f"Error processing message: {e}"
            self.log_to_file(message)
            with self.performance_lock:
                self.messages_failed += 1
    
    def handle_sensor_data(self, topic, data):
        """Handle sensor data using DQN decision making"""
        try:
            sensor_id = data.get('sensor_id', 'unknown')
            sensor_type = data.get('sensor_type', 'unknown')
            location = data.get('location', 'unknown')
            value = data.get('value')
            timestamp = data.get('timestamp', 'unknown')
            unit = data.get('unit', 'unknown')
            sensor_bat = data.get('sensor_bat', 100)
            
            current_time = time.time()
            deadline = self.sensor_deadlines.get(sensor_type, 60)
            
            # Calculate time since last transmission
            last_time = self.sensor_last_received.get(sensor_id, datetime.now() - timedelta(seconds=deadline))
            time_since_last = (datetime.now() - last_time).total_seconds()
            remaining_time = max(deadline - time_since_last, 0)
            
            # Emergency override for deadline approaching
            if time_since_last > (deadline / 2):
                action = 1  # Force transmit
                decision_type = "[FORCED TRANSMIT]"
                self.log_to_file(f"{decision_type} Deadline emergency: {sensor_id}")
            else:
                # Track delay statistics
                processing_delay = time_since_last
                self.delay_stats['all'].append(processing_delay)
                self.delay_stats['by_sensor_type'][sensor_type].append(processing_delay)
                self.delay_stats['by_location'][location].append(processing_delay)
                
                # Check if critical value
                is_critical = self.is_critical_value(sensor_type, value)
                if is_critical:
                    self.critical_sensors_count += 1
                    self.delay_stats['critical_values'].append(processing_delay)
                
                # Generate state vector
                state = self.get_state(sensor_type, location, current_time, time_since_last, remaining_time, value, sensor_bat)
                
                # Get DQN decision
                action = self.dqn_agent.act(
                    state, 
                    self.sensor_priority.get(sensor_type, 1.0), 
                    self.location_priority.get(location, 1.0)
                )
                
                # Calculate reward for training
                reward = self.calculate_reward(action, sensor_type, location, remaining_time, value, sensor_bat)
                
                # Store experience for training (simplified - using current state as next_state)
                done = remaining_time <= 0
                self.dqn_agent.remember(state, action, reward, state, done)
                
                # Train the agent
                if len(self.dqn_agent.memory) > self.dqn_agent.batch_size:
                    self.dqn_agent.replay()
                
                decision_type = "[DQN]"
            
            # Update tracking
            current_time_dt = datetime.now()
            with self.performance_lock:
                self.sensor_last_received[sensor_id] = current_time_dt
                if sensor_type in self.sensor_total_reads:
                    self.sensor_total_reads[sensor_type] += 1
                    if action == 1:  # Transmitted
                        self.sensor_successful_transfers[sensor_type] += 1
                        self.sensors_processed_count += 1
            
            # Log decision (only to file)
            decision_text = "TRANSMIT" if action == 1 else "DROP"
            critical_status = "CRITICAL" if self.is_critical_value(sensor_type, value) else "NORMAL"
            battery_status = "LOW" if sensor_bat < 15 else "MED" if sensor_bat < 50 else "HIGH"
            
            message = f"{decision_type} {decision_text} - [{critical_status}] [{battery_status} BAT] Sensor: {sensor_id}, Type: {sensor_type}, Location: {location}, Value: {value}, Battery: {sensor_bat}%, Delay: {time_since_last:.2f}s"
            self.log_to_file(message)
            
            # Process sensor data if decision is to transmit
            if action == 1:
                if sensor_type == "heart_rate":
                    self.handle_heart_rate_data(sensor_id, location, value, timestamp, unit, sensor_bat)
                elif sensor_type == "body_temperature":
                    self.handle_body_temperature_data(sensor_id, location, value, timestamp, unit, sensor_bat)
                elif sensor_type == "spo2":
                    self.handle_spo2_data(sensor_id, location, value, timestamp, unit, sensor_bat)
                elif sensor_type == "blood_pressure":
                    self.handle_blood_pressure_data(sensor_id, location, value, timestamp, unit, sensor_bat)
                elif sensor_type == "respiratory_rate":
                    self.handle_respiratory_rate_data(sensor_id, location, value, timestamp, unit, sensor_bat)
                elif sensor_type == "ambient_temperature":
                    self.handle_ambient_temperature_data(sensor_id, location, value, timestamp, unit, sensor_bat)
                elif sensor_type == "humidity":
                    self.handle_humidity_data(sensor_id, location, value, timestamp, unit, sensor_bat)
                
        except Exception as e:
            message = f"Error handling sensor data: {e}"
            self.log_to_file(message)
    
    def handle_heart_rate_data(self, sensor_id, location, value, timestamp, unit, sensor_bat):
        """Handle heart rate sensor data"""
        try:
            heart_rate = int(value)
            
            message = f"Heart Rate - Sensor: {sensor_id}, Location: {location}, Value: {heart_rate} {unit}, Battery: {sensor_bat}%"
            self.log_to_file(message)
            
            if self.puf.is_ready():
                encoded_key = self.puf.get_heart_rate_key(heart_rate)
                if encoded_key:
                    message = f"Encoded heart rate key: {encoded_key}"
                    self.log_to_file(message)
                else:
                    message = "Failed to encode heart rate"
                    self.log_to_file(message)
            else:
                message = "PUF system not ready"
                self.log_to_file(message)
                
        except (ValueError, TypeError):
            message = f"Invalid heart rate format: {value}"
            self.log_to_file(message)
    
    def handle_body_temperature_data(self, sensor_id, location, value, timestamp, unit, sensor_bat):
        """Handle body temperature sensor data"""
        try:
            temperature = float(value)
            
            message = f"Body Temperature - Sensor: {sensor_id}, Location: {location}, Value: {temperature} {unit}, Battery: {sensor_bat}%"
            self.log_to_file(message)
            
            if self.puf.is_ready():
                encoded_key = self.puf.get_temperature_key(temperature)
                if encoded_key:
                    message = f"Encoded temperature key: {encoded_key}"
                    self.log_to_file(message)
                else:
                    message = "Failed to encode temperature"
                    self.log_to_file(message)
            else:
                message = "PUF system not ready"
                self.log_to_file(message)
                
        except (ValueError, TypeError):
            message = f"Invalid temperature format: {value}"
            self.log_to_file(message)
    
    def handle_spo2_data(self, sensor_id, location, value, timestamp, unit, sensor_bat):
        """Handle SpO2 sensor data"""
        try:
            spo2_value = float(value)
            
            message = f"SpO2 - Sensor: {sensor_id}, Location: {location}, Value: {spo2_value} {unit}, Battery: {sensor_bat}%"
            self.log_to_file(message)
            
            message = f"SpO2 data logged - no PUF encoding implemented for this sensor type"
            self.log_to_file(message)
            
        except (ValueError, TypeError):
            message = f"Invalid SpO2 format: {value}"
            self.log_to_file(message)
    
    def handle_blood_pressure_data(self, sensor_id, location, value, timestamp, unit, sensor_bat):
        """Handle blood pressure sensor data"""
        try:
            if isinstance(value, dict):
                systolic = value.get('systolic', 0)
                diastolic = value.get('diastolic', 0)
                
                message = f"Blood Pressure - Sensor: {sensor_id}, Location: {location}, Systolic: {systolic} {unit}, Diastolic: {diastolic} {unit}, Battery: {sensor_bat}%"
                self.log_to_file(message)
                
                message = f"Blood pressure data logged - no PUF encoding implemented for this sensor type"
                self.log_to_file(message)
            else:
                message = f"Invalid blood pressure format - expected dict with systolic/diastolic: {value}"
                self.log_to_file(message)
                
        except Exception as e:
            message = f"Error processing blood pressure data: {e}"
            self.log_to_file(message)
    
    def handle_respiratory_rate_data(self, sensor_id, location, value, timestamp, unit, sensor_bat):
        """Handle respiratory rate sensor data"""
        try:
            resp_rate = int(value)
            
            message = f"Respiratory Rate - Sensor: {sensor_id}, Location: {location}, Value: {resp_rate} {unit}, Battery: {sensor_bat}%"
            self.log_to_file(message)
            
            message = f"Respiratory rate data logged - no PUF encoding implemented for this sensor type"
            self.log_to_file(message)
            
        except (ValueError, TypeError):
            message = f"Invalid respiratory rate format: {value}"
            self.log_to_file(message)
    
    def handle_ambient_temperature_data(self, sensor_id, location, value, timestamp, unit, sensor_bat):
        """Handle ambient temperature sensor data"""
        try:
            ambient_temp = float(value)
            
            message = f"Ambient Temperature - Sensor: {sensor_id}, Location: {location}, Value: {ambient_temp} {unit}, Battery: {sensor_bat}%"
            self.log_to_file(message)
            
            message = f"Ambient temperature data logged - no PUF encoding implemented for this sensor type"
            self.log_to_file(message)
            
        except (ValueError, TypeError):
            message = f"Invalid ambient temperature format: {value}"
            self.log_to_file(message)
    
    def handle_humidity_data(self, sensor_id, location, value, timestamp, unit, sensor_bat):
        """Handle humidity sensor data"""
        try:
            humidity = float(value)
            
            message = f"Humidity - Sensor: {sensor_id}, Location: {location}, Value: {humidity} {unit}, Battery: {sensor_bat}%"
            self.log_to_file(message)
            
            message = f"Humidity data logged - no PUF encoding implemented for this sensor type"
            self.log_to_file(message)
            
        except (ValueError, TypeError):
            message = f"Invalid humidity format: {value}"
            self.log_to_file(message)
    
    def handle_legacy_format(self, topic, payload):
        """Handle legacy format for backward compatibility"""
        if "temperature" in topic:
            self.handle_temperature(payload)
        elif "heart_rate" in topic:
            self.handle_heart_rate(payload)
        else:
            message = f"Unknown legacy topic: {topic}"
            self.log_to_file(message)
    
    def handle_temperature(self, payload):
        """Handle legacy temperature format"""
        try:
            data = json.loads(payload)
            temperature = float(data.get('value', 0))
            
            message = f"Temperature received (legacy): {temperature}°C"
            self.log_to_file(message)
            
            if self.puf.is_ready():
                encoded_key = self.puf.get_temperature_key(temperature)
                if encoded_key:
                    message = f"Encoded temperature key: {encoded_key}"
                    self.log_to_file(message)
                else:
                    message = "Failed to encode temperature"
                    self.log_to_file(message)
            else:
                message = "PUF system not ready"
                self.log_to_file(message)
                
        except json.JSONDecodeError:
            try:
                temperature = float(payload)
                message = f"Temperature received (legacy): {temperature}°C"
                self.log_to_file(message)
                
                if self.puf.is_ready():
                    encoded_key = self.puf.get_temperature_key(temperature)
                    if encoded_key:
                        message = f"Encoded temperature key: {encoded_key}"
                        self.log_to_file(message)
                    else:
                        message = "Failed to encode temperature"
                        self.log_to_file(message)
                else:
                    message = "PUF system not ready"
                    self.log_to_file(message)
            except ValueError:
                message = f"Invalid temperature format: {payload}"
                self.log_to_file(message)
    
    def handle_heart_rate(self, payload):
        """Handle legacy heart rate format"""
        try:
            data = json.loads(payload)
            heart_rate = int(data.get('bpm', 0))
            
            message = f"Heart rate received (legacy): {heart_rate} BPM"
            self.log_to_file(message)
            
            if self.puf.is_ready():
                encoded_key = self.puf.get_heart_rate_key(heart_rate)
                if encoded_key:
                    message = f"Encoded heart rate key: {encoded_key}"
                    self.log_to_file(message)
                else:
                    message = "Failed to encode heart rate"
                    self.log_to_file(message)
            else:
                message = "PUF system not ready"
                self.log_to_file(message)
                
        except json.JSONDecodeError:
            try:
                heart_rate = int(payload)
                message = f"Heart rate received (legacy): {heart_rate} BPM"
                self.log_to_file(message)
                
                if self.puf.is_ready():
                    encoded_key = self.puf.get_heart_rate_key(heart_rate)
                    if encoded_key:
                        message = f"Encoded heart rate key: {encoded_key}"
                        self.log_to_file(message)
                    else:
                        message = "Failed to encode heart rate"
                        self.log_to_file(message)
                else:
                    message = "PUF system not ready"
                    self.log_to_file(message)
            except ValueError:
                message = f"Invalid heart rate format: {payload}"
                self.log_to_file(message)
    
    def connect_mqtt(self, broker_host="localhost", broker_port=1883):
        try:
            self.client.connect(broker_host, broker_port, 60)
            return True
        except Exception as e:
            message = f"Failed to connect to MQTT broker: {e}"
            self.log_to_file(message)
            return False
    
    def get_max_processing_rate(self):
        """Get maximum processing rate achieved"""
        with self.performance_lock:
            return self.max_processing_rate
    
    def start_listening(self):
        try:
            self.client.loop_start()
            while not self.shutdown_flag.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown_gracefully()

def select_serial_port():
    """Show port list and let user select one"""
    ports = get_available_serial_ports()
    if not ports:
        print("No serial ports found!")
        return None
    
    print("Available serial ports:")
    for i, (device, description) in enumerate(ports):
        print(f"{i+1}. {device} - {description}")
    
    while True:
        choice = input(f"\nSelect port number (1-{len(ports)}): ").strip()
        
        try:
            choice = int(choice)
            if 1 <= choice <= len(ports):
                selected_port = ports[choice-1][0]
                print(f"Selected port: {selected_port}")
                return selected_port
            else:
                print(f"Invalid choice. Please enter 1-{len(ports)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    print("MQTT Sensor Data Reader with PUF Obfuscation - DQN Priority System (Silent Mode - Optimized)")
    print("Date: 2025-09-14")
    print("User: foladlo")
    print("="*60)
    
    selected_port = select_serial_port()
    if not selected_port:
        print("No port selected. Exiting...")
        return
    
    from puf_interface import PUFInterface, create_puf_interface
    
    puf = create_puf_interface(selected_port)
    
    print("Initializing PUF system...")
    if puf.initialize():
        if puf.is_ready():
            print("PUF system ready")
            print(f"Temperature table size: {puf.get_temp_table_size()}")
            print(f"Heart rate table size: {puf.get_hr_table_size()}")
        else:
            print("PUF system not ready")
            return
    else:
        print("Failed to initialize PUF system")
        return
    
    mqtt_reader = MQTTSensorReader(puf)
    
    mqtt_reader.log_to_file("MQTT Sensor Data Reader with DQN Priority System (Silent Mode - Optimized) started")
    mqtt_reader.log_to_file(f"Selected port: {selected_port}")
    mqtt_reader.log_to_file(f"Temperature table size: {puf.get_temp_table_size()}")
    mqtt_reader.log_to_file(f"Heart rate table size: {puf.get_hr_table_size()}")
    
    print(f"Log file created: {mqtt_reader.log_file}")
    print(f"Performance log file created: {mqtt_reader.performance_file}")
    
    broker_host = input("Enter MQTT broker host (default: localhost): ").strip()
    if not broker_host:
        broker_host = "localhost"
    
    try:
        broker_port = input("Enter MQTT broker port (default: 1883): ").strip()
        if not broker_port:
            broker_port = 1883
        else:
            broker_port = int(broker_port)
    except ValueError:
        broker_port = 1883
    
    if mqtt_reader.connect_mqtt(broker_host, broker_port):
        print("Connected to MQTT broker successfully!")
        print("\nDQN Priority System Active (Silent Mode - Optimized):")
        print("- PyTorch tensor optimization applied")
        print("- Reduced monitoring frequencies")
        print("- Emergency deadline override implemented")
        print("- Smaller batch sizes for faster training")
        print("- Press Ctrl+C to stop and save DQN model...")
        
        mqtt_reader.log_to_file("Connected to MQTT broker successfully with DQN priority system (Silent Mode - Optimized)!")
        
        try:
            mqtt_reader.start_listening()
        except KeyboardInterrupt:
            mqtt_reader.shutdown_gracefully()
    else:
        print("Failed to connect to MQTT broker")
        if puf:
            puf.disconnect()

if __name__ == "__main__":
    main()