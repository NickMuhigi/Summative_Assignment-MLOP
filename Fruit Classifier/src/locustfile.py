"""
Locust load testing script for the Fruit Classifier API
Run with: locust -f locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, task, between
import random
import io
from PIL import Image
import numpy as np

class FruitClassifierUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Create test images when user starts"""
        self.test_images = self.create_test_images()
    
    def create_test_images(self):
        """Create sample test images in memory"""
        images = []
        
        # Create 5 different test images
        for i in range(5):
            # Create a random 64x64 RGB image
            img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            images.append({
                'name': f'test_image_{i}.jpg',
                'data': img_bytes.getvalue(),
                'content_type': 'image/jpeg'
            })
        
        return images
    
    @task(10)  # Weight: 10 (most common task)
    def predict_fruit(self):
        """Test the prediction endpoint"""
        # Select a random test image
        test_image = random.choice(self.test_images)
        
        files = {
            'file': (test_image['name'], test_image['data'], test_image['content_type'])
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'prediction' in result:
                    response.success()
                else:
                    response.failure("No prediction in response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(3)  # Weight: 3
    def check_status(self):
        """Test the status endpoint"""
        with self.client.get("/status", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'api_status' in result:
                    response.success()
                else:
                    response.failure("Invalid status response")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(2)  # Weight: 2
    def health_check(self):
        """Test the health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'healthy':
                    response.success()
                else:
                    response.failure("Unhealthy status")
            else:
                response.failure(f"Got status code {response.status_code}")
    
    @task(1)  # Weight: 1 (least common)
    def root_endpoint(self):
        """Test the root endpoint"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                result = response.json()
                if 'message' in result:
                    response.success()
                else:
                    response.failure("Invalid root response")
            else:
                response.failure(f"Got status code {response.status_code}")

class HeavyUser(HttpUser):
    """Simulates heavy usage patterns"""
    wait_time = between(0.1, 0.5)  # Very frequent requests
    weight = 1  # Lower weight, fewer of these users
    
    def on_start(self):
        self.test_images = self.create_test_images()
    
    def create_test_images(self):
        """Create larger test images to stress the system"""
        images = []
        
        for i in range(3):
            # Create larger images (128x128)
            img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            images.append({
                'name': f'heavy_test_{i}.jpg',
                'data': img_bytes.getvalue(),
                'content_type': 'image/jpeg'
            })
        
        return images
    
    @task
    def rapid_predictions(self):
        """Make rapid predictions to stress test"""
        test_image = random.choice(self.test_images)
        
        files = {
            'file': (test_image['name'], test_image['data'], test_image['content_type'])
        }
        
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}")

# Custom load testing scenarios
class BurstUser(HttpUser):
    """Simulates burst traffic patterns"""
    wait_time = between(5, 10)  # Long wait, then burst
    weight = 1
    
    def on_start(self):
        # Create one test image
        img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        self.test_image = {
            'name': 'burst_test.jpg',
            'data': img_bytes.getvalue(),
            'content_type': 'image/jpeg'
        }
    
    @task
    def burst_requests(self):
        """Make multiple rapid requests in a burst"""
        for i in range(5):  # 5 rapid requests
            files = {
                'file': (f'burst_{i}.jpg', self.test_image['data'], self.test_image['content_type'])
            }
            
            with self.client.post("/predict", files=files, catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"Burst request {i} failed")
                else:
                    response.success()
            
            # Small delay between burst requests
            self.wait_time = between(0.1, 0.2)