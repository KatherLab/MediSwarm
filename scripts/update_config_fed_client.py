#!/usr/bin/env python3
"""
Utility to update config_fed_client.conf for different challenge models.
"""

import sys
import re
from pathlib import Path
from typing import Dict, Optional
import logging

# Import shared model configurations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'application', 'jobs', 'ODELIA_ternary_classification', 'app', 'custom', 'models'))
from challenge.challenge_models_config import CHALLENGE_MODELS, get_model_config, get_persistor_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use shared configurations
MODEL_CONFIGS = CHALLENGE_MODELS


class ConfigFedClientUpdater:
    """Update config_fed_client.conf with model-specific persistor configuration."""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    def update_model(self, model_name: str, backup: bool = True) -> bool:
        """
        Update config for the specified model.
        
        Args:
            model_name: Name of the challenge model (1DivideAndConquer, 2BCN_AIM, etc.)
            backup: Create backup of original config before updating
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            logger.error(f"Available models: {', '.join(MODEL_CONFIGS.keys())}")
            return False
        
        model_config = MODEL_CONFIGS[model_name]
        
        # Create backup
        if backup:
            backup_path = self.config_path.with_suffix(f".{model_name}.backup")
            with open(self.config_path, 'r') as f:
                backup_content = f.read()
            with open(backup_path, 'w') as f:
                f.write(backup_content)
            logger.info(f"Created backup: {backup_path}")
        
        # Read current config
        with open(self.config_path, 'r') as f:
            config_content = f.read()
        
        # Generate new persistor section
        new_persistor = self._generate_persistor_section(
            model_config["persistor_path"],
            model_config["persistor_args"]
        )
        
        # Replace persistor component using regex
        # Find the persistor block and replace it
        pattern = (
            r'(\{\s*'
            r'id\s*=\s*"persistor"\s*'
            r'path\s*=\s*"nvflare\.app_opt\.pt\.file_model_persistor\.PTFileModelPersistor"\s*'
            r'args\s*\{[^}]*model\s*\{[^}]*\}[^}]*\}'
            r'\s*\})'
        )
        
        # Try more lenient pattern
        pattern = (
            r'(\{\s*id\s*=\s*"persistor".*?'
            r'(?=\s*\{\s*id\s*=\s*"shareable_generator")'
        )
        
        updated_config = re.sub(
            pattern,
            new_persistor + '\n  }',
            config_content,
            flags=re.DOTALL
        )
        
        # If regex didn't work, try simpler approach
        if updated_config == config_content:
            logger.warning("Could not update using regex, trying manual approach...")
            updated_config = self._update_config_manual(config_content, new_persistor)
        
        # Write updated config
        with open(self.config_path, 'w') as f:
            f.write(updated_config)
        
        logger.info(f"Updated config for model: {model_name}")
        logger.info(f"Persistor path: {model_config['persistor_path']}")
        return True
    
    def _generate_persistor_section(self, persistor_path: str, args: Dict) -> str:
        """Generate persistor component section in HOCON format."""
        args_lines = []
        for key, value in args.items():
            if isinstance(value, str):
                args_lines.append(f'        {key}="{value}"')
            else:
                args_lines.append(f'        {key}={value}')
        
        args_block = "\n".join(args_lines)
        
        return f'''  {{
    id = "persistor"
    path = "nvflare.app_opt.pt.file_model_persistor.PTFileModelPersistor"
    args {{
      model {{
        path = "{persistor_path}"
        args {{
{args_block}
        }}
      }}
    }}
  }}'''
    
    def _update_config_manual(self, config_content: str, new_persistor: str) -> str:
        """Manual approach to update config by replacing persistor block."""
        lines = config_content.split('\n')
        in_persistor = False
        brace_count = 0
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(lines):
            if 'id = "persistor"' in line:
                in_persistor = True
                start_idx = self._find_block_start(lines, i)
            
            if in_persistor:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0 and start_idx != -1 and '{' in line:
                    end_idx = i + 1
                    break
        
        if start_idx != -1 and end_idx != -1:
            new_lines = lines[:start_idx] + new_persistor.split('\n') + lines[end_idx:]
            return '\n'.join(new_lines)
        
        logger.error("Could not find persistor block in config")
        return config_content
    
    def _find_block_start(self, lines, current_idx):
        """Find the opening brace of the block containing current line."""
        for i in range(current_idx, -1, -1):
            if '{' in lines[i]:
                return i
        return current_idx
    
    def list_models(self):
        """List available models."""
        print("Available challenge models:")
        for model_name, config in MODEL_CONFIGS.items():
            print(f"\n  {model_name}:")
            print(f"    Path: {config['persistor_path']}")
            print(f"    Args: {config['persistor_args']}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python update_config.py <model_name> [config_path]")
        print("\nExample:")
        print("  python update_config.py 2BCN_AIM")
        print("  python update_config.py 3agaldran /path/to/config_fed_client.conf")
        print("\nAvailable models:")
        updater = ConfigFedClientUpdater(
            "/home/swarm/Documents/MediSwarmChallenge/MediSwarm/"
            "application/jobs/ODELIA_ternary_classification/app/config/config_fed_client.conf"
        )
        updater.list_models()
        return
    
    model_name = sys.argv[1]
    
    if model_name == "--list":
        updater = ConfigFedClientUpdater(
            "/home/swarm/Documents/MediSwarmChallenge/MediSwarm/"
            "application/jobs/ODELIA_ternary_classification/app/config/config_fed_client.conf"
        )
        updater.list_models()
        return
    
    config_path = sys.argv[2] if len(sys.argv) > 2 else (
        "/home/swarm/Documents/MediSwarmChallenge/MediSwarm/"
        "application/jobs/ODELIA_ternary_classification/app/config/config_fed_client.conf"
    )
    
    try:
        updater = ConfigFedClientUpdater(config_path)
        success = updater.update_model(model_name)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
