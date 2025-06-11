import time
import logging
from functools import wraps
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='performance.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('performance_metrics')

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {}
        
    @staticmethod
    def log_time(name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                execution_time = end_time - start_time
                logger.info(f"{name} took {execution_time:.2f} seconds")
                
                # Store metrics
                timestamp = datetime.now().isoformat()
                if not hasattr(wrapper, 'metrics'):
                    wrapper.metrics = []
                wrapper.metrics.append({
                    'timestamp': timestamp,
                    'execution_time': execution_time
                })
                
                # Save metrics to file periodically
                if len(wrapper.metrics) >= 10:
                    with open(f'metrics_{name}.json', 'a') as f:
                        json.dump(wrapper.metrics, f)
                        wrapper.metrics = []
                
                return result
            return wrapper
        return decorator

performance_metrics = PerformanceMetrics() 