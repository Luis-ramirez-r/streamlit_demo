import asyncio
import aiohttp
import json
from datetime import datetime
import numpy as np
import tomli
import pandas as pd
from typing import Dict, List
import random
from pathlib import Path

class EngineSimulator:
    def __init__(self, config_path: str = 'config.toml'):
        # Load configuration
        with open(config_path, 'rb') as f:
            self.config = tomli.load(f)
        
        # Load training data
        data_path = Path(self.config['paths']['interim_data']) / 'train_FD001.parquet'
        self.data = pd.read_parquet(data_path)
        
        # Convert engine_id to integer
        self.data['engine_id'] = self.data['engine_id'].astype(int)
        
        # Get list of unique engine IDs
        self.engine_ids = self.data['engine_id'].unique()
        
        # API endpoint
        self.api_url = "http://localhost:8505/predict"
        
        # Store the sensor columns
        self.sensor_columns = [col for col in self.data.columns 
                             if col not in ['engine_id', 'cycle', 'RUL']]
    
    def get_engine_lifecycle(self, engine_id: int = None) -> pd.DataFrame:
        """Get complete lifecycle data for a random or specific engine."""
        if engine_id is None:
            engine_id = random.choice(self.engine_ids)
        
        lifecycle_data = self.data[self.data['engine_id'] == engine_id].copy()
        lifecycle_data = lifecycle_data.sort_values('cycle').reset_index(drop=True)
        return lifecycle_data
    
    def generate_reading(self, lifecycle_data: pd.DataFrame, cycle_idx: int) -> Dict:
        """Generate a sensor reading from actual data."""
        reading = lifecycle_data.iloc[cycle_idx]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "engine_id": f"ENG{int(reading['engine_id']):03d}",
            "readings": {
                column: float(reading[column])
                for column in self.sensor_columns
            }
        }

    async def send_reading(self, session: aiohttp.ClientSession, reading: Dict):
        """Send a reading to the FastAPI endpoint."""
        try:
            async with session.post(self.api_url, json=reading) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    print(f"Error from API: {error_text}")
                    return None
        except Exception as e:
            print(f"Error sending reading: {e}")
            return None

    async def simulate_engine(self, engine_id: int = None):
        """Simulate an engine's lifecycle using real data."""
        try:
            # Get complete lifecycle data for an engine
            lifecycle_data = self.get_engine_lifecycle(engine_id)
            total_cycles = len(lifecycle_data)
            
            engine_id_str = f"ENG{int(lifecycle_data['engine_id'].iloc[0]):03d}"
            print(f"Starting simulation for {engine_id_str}")
            print(f"Total cycles: {total_cycles}")
            
            async with aiohttp.ClientSession() as session:
                for cycle in range(total_cycles):
                    reading = self.generate_reading(lifecycle_data, cycle)
                    
                    result = await self.send_reading(session, reading)
                    if result:
                        print(f"{engine_id_str} - Cycle {cycle}/{total_cycles}: "
                              f"{result['predicted_label']}")
                    
                    await asyncio.sleep(1)  # 1 second between readings
        except Exception as e:
            print(f"Error in engine simulation: {e}")

class SimulationManager:
    def __init__(self, num_engines: int = 3):
        self.simulator = EngineSimulator()
        self.num_engines = num_engines
    
    async def run_simulation(self):
        """Run simulation for multiple engines."""
        try:
            # Randomly select engines for simulation
            selected_engines = random.sample(
                list(self.simulator.engine_ids), 
                min(self.num_engines, len(self.simulator.engine_ids))
            )
            
            print(f"Selected engines for simulation: {selected_engines}")
            
            # Create tasks for each engine
            tasks = [self.simulator.simulate_engine(engine_id) 
                    for engine_id in selected_engines]
            
            # Run simulations concurrently
            await asyncio.gather(*tasks)
            
        except Exception as e:
            print(f"Error in simulation manager: {e}")

async def main():
    try:
        # Initialize and run simulation
        manager = SimulationManager(num_engines=3)
        await manager.run_simulation()
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())