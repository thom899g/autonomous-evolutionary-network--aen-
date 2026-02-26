"""
Autonomous Evolutionary Node - Core Orchestrator
Coordinates Perception and Cognition brains with real-time market validation
"""
import asyncio
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from loguru import logger
from pydantic import BaseModel, Field
import traceback

from .perception_brain import PerceptionBrain, FitnessMetrics
from .cognition_brain import CognitionBrain, LearningState
from .memory_palace import MemoryPalace, NetworkState
from .market_validator import MarketValidator

# Configure structured logging
logger.add("logs/node_{time}.log", rotation="500 MB", retention="10 days")

class NodeConfig(BaseModel):
    """Validated node configuration"""
    node_id: str = Field(..., min_length=3, max_length=50)
    node_type: str = Field(..., regex="^(arbitrage_hunter|volatility_sponge|correlation_mapper)$")
    network_region: str = "us-west-2"
    enable_market_validation: bool = True
    sync_interval_seconds: int = 30
    max_retry_attempts: int = 3

@dataclass
class NodeState:
    """Immutable node state snapshot"""
    timestamp: float
    fitness_score: float
    strategy_embedding: Dict[str, Any]
    market_conditions: Dict[str, float]
    network_coherence: float
    memory_usage_mb: float

class AENNode:
    """Main node orchestrator implementing dual-brain architecture"""
    
    def __init__(self, config: NodeConfig):
        """Initialize node with validated configuration"""
        self.config = config
        self.state: Optional[NodeState] = None
        self.running = False
        
        # Initialize core components with error handling
        try:
            logger.info(f"Initializing node {config.node_id} of type {config.node_type}")
            
            self.memory = MemoryPalace(node_id=config.node_id)
            self.perception = PerceptionBrain(node_type=config.node_type)
            self.cognition = CognitionBrain(memory_palace=self.memory)
            
            if config.enable_market_validation:
                self.validator = MarketValidator()
                logger.success("Market validation enabled")
            else:
                self.validator = None
                logger.warning("Market validation disabled - running in simulation mode")
                
            self._initialize_firebase()
            logger.success(f"Node {config.node_id} initialized successfully")
            
        except Exception as e:
            logger.error(f"Node initialization failed: {e}")
            self._emergency_telegram_alert(f"Node {config.node_id} failed to initialize: {str(e)}")
            raise
    
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection with fallback mechanisms"""
        try:
            # Firestore will be initialized in MemoryPalace
            logger.debug("Firebase initialization delegated to MemoryPalace")
        except ImportError as e:
            logger.critical(f"Firebase not available: {e}")
            raise
    
    async def start(self) -> None:
        """Start autonomous evolutionary cycle"""
        self.running = True
        logger.info(f"Starting evolutionary cycle for node {self.config.node_id}")
        
        # Initial synchronization with network
        await self._sync_with_network()
        
        # Main evolutionary loop
        while self.running:
            try:
                cycle_start = asyncio.get_event_loop().time()
                
                # Phase 1: Generate and validate strategy
                strategy = await self._generate_strategy()
                validated = await self._validate_strategy(strategy)
                
                if validated:
                    # Phase 2: Update cognition and memory
                    learning_state = await self._update_cognition(strategy)
                    
                    # Phase 3: Synchronize with network
                    await self._sync_with_network()
                    
                    # Phase 4: Update node state
                    await self._update_state(strategy, learning_state)
                
                # Calculate cycle time and adjust if needed
                cycle_time = asyncio.get_event_loop().time() - cycle_start
                if cycle_time < self.config.sync_interval_seconds:
                    await asyncio.sleep(self.config.sync_interval_seconds - cycle_time)
                    
            except asyncio.CancelledError:
                logger.info(f"Node {self.config.node_id} received shutdown signal")
                break
            except Exception as e:
                logger.error(f"Evolutionary cycle error: {e}")
                await self._handle_cycle_error(e)
    
    async def _generate_strategy(self) -> Dict[str, Any]:
        """Generate new strategy using perception brain"""
        try:
            # Get current market context
            market_context = await self._get_market_context()
            
            # Generate strategy embedding
            strategy = self.perception.generate_strategy(
                market_context=market_context,
                fitness_metrics=await self._get_fitness_metrics()
            )
            
            logger.debug(f"Generated strategy with hash: {hash(str(strategy))}")
            return strategy