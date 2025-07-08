#!/usr/bin/env python3
"""
Dataset Generation Tool for IBVS Neural Network Training
"""
import matplotlib
matplotlib.use('Agg')

import argparse
import logging
import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.simulation import SimulationEngine
from core.data_handler import DatasetWriter
from config.settings import DataConfig, Paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Main dataset generation class"""
    
    def __init__(self, output_dir: str = None, 
                 max_iterations: int = DataConfig.MAX_ITERATIONS,
                 lambda_value: float = DataConfig.LAMBDA_VALUE,
                 batch_size: int = DataConfig.SAVE_BATCH_SIZE):
        
        self.output_dir = output_dir or Paths.DATA_DIR
        self.max_iterations = max_iterations
        self.lambda_value = lambda_value
        self.batch_size = batch_size
        
        self.simulation_engine = SimulationEngine(
            max_iterations=max_iterations,
            lambda_gain=lambda_value,
            convergence_threshold=DataConfig.CONVERGENCE_THRESHOLD
        )
        
        self.data_writer = DatasetWriter(
            os.path.join(self.output_dir, "training_data")
        )
        
        logger.info(f"Dataset generator initialized")
        logger.info(f"Output directory: {self.output_dir}")
    
    def generate_single_sequence(self, sequence_id: int) -> int:
        """Generate a single training sequence"""
        try:
            self.simulation_engine.prepare_simulation()
            data_points = self.simulation_engine.run_single_simulation(sequence_id)
            self.data_writer.add_data_points(data_points)
            
            logger.debug(f"Generated sequence {sequence_id} with {len(data_points)} points")
            return len(data_points)
            
        except Exception as e:
            logger.error(f"Error generating sequence {sequence_id}: {e}")
            return 0
    
    def generate_batch_sequences(self, num_sequences: int, start_id: int = 0) -> dict:
        """Generate multiple training sequences"""
        logger.info(f"Starting batch generation of {num_sequences} sequences")
        
        total_points = 0
        successful_sequences = 0
        failed_sequences = 0
        
        for i in range(num_sequences):
            sequence_id = start_id + i
            
            points_count = self.generate_single_sequence(sequence_id)
            
            if points_count > 0:
                successful_sequences += 1
                total_points += points_count
            else:
                failed_sequences += 1
            
            if (i + 1) % self.batch_size == 0:
                self.data_writer.save_batch(self.batch_size)
                logger.info(f"Saved batch after {i + 1} sequences")
            
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{num_sequences} sequences generated")
        
        self.data_writer.save_all()
        
        stats = {
            'total_sequences': num_sequences,
            'successful_sequences': successful_sequences,
            'failed_sequences': failed_sequences,
            'total_data_points': total_points,
            'avg_points_per_sequence': total_points / successful_sequences if successful_sequences > 0 else 0
        }
        
        logger.info(f"Batch generation completed:")
        logger.info(f"  - Successful sequences: {successful_sequences}")
        logger.info(f"  - Failed sequences: {failed_sequences}")
        logger.info(f"  - Total data points: {total_points}")
        
        return stats

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate IBVS training dataset')
    
    parser.add_argument('--num-sequences', type=int, default=1000,
                        help='Number of sequences to generate (default: 1000)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for dataset files')
    parser.add_argument('--max-iterations', type=int, default=DataConfig.MAX_ITERATIONS,
                        help='Maximum iterations per sequence')
    parser.add_argument('--lambda-value', type=float, default=DataConfig.LAMBDA_VALUE,
                        help='Lambda gain value for IBVS controller')
    parser.add_argument('--batch-size', type=int, default=DataConfig.SAVE_BATCH_SIZE,
                        help='Batch size for saving data')
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    
    logger.info("Starting dataset generation...")
    logger.info(f"Parameters: {args}")
    
    try:
        generator = DatasetGenerator(
            output_dir=args.output_dir,
            max_iterations=args.max_iterations,
            lambda_value=args.lambda_value,
            batch_size=args.batch_size
        )
        
        stats = generator.generate_batch_sequences(args.num_sequences)
        
        logger.info("Dataset generation completed successfully")
        logger.info(f"Final statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 