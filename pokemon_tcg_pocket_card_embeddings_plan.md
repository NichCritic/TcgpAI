# Pokemon TCG Pocket MBRL - Core Learning Loop

## Minimal Viable System

Get a basic model-based RL agent learning Pokemon TCG Pocket in 4 core components:

## 1. Data Collection

**What we need:**
- Game states: `{cards_in_hand, cards_on_field, energy_count, hp_values, turn_phase}`
- Actions: `{play_card, attack, use_ability, end_turn}`  
- Outcomes: `{next_state, reward, done}`

**Implementation:**
```python
class GameState:
    def __init__(self):
        self.my_hand = []           # List of card objects
        self.my_field = []          # Active + bench Pokemon
        self.opponent_field = []    # Visible opponent cards
        self.my_energy = 0          # Energy count
        self.turn_phase = "main"    # "setup", "main", "attack"
        self.rewards = 0            # Damage dealt, prizes taken

class DataCollector:
    def collect_game(self):
        transitions = []
        while not game.done():
            state = game.get_state()
            action = get_action()  # Random or human
            next_state, reward = game.step(action)
            transitions.append((state, action, reward, next_state))
        return transitions
```

## 2. Unified Card-State Encoder

**Core insight:** Card value depends entirely on current game context.

```python
class CardStateEncoder(nn.Module):
    def __init__(self):
        # Card properties: [type, hp, attack_damage, energy_cost]
        self.card_encoder = nn.Linear(card_features, 128)
        
        # Game context: [energy_available, turn_phase, field_state]
        self.context_encoder = nn.Linear(context_features, 128)
        
        # Fusion
        self.fusion = nn.MultiheadAttention(128, 4)
        
    def forward(self, cards, game_state):
        card_embeds = self.card_encoder(cards)
        context_embed = self.context_encoder(game_state)
        
        # Each card attends to game context
        fused_cards, _ = self.fusion(card_embeds, context_embed, context_embed)
        return fused_cards

class StateEncoder(nn.Module):
    def __init__(self):
        self.card_state_encoder = CardStateEncoder()
        self.global_encoder = nn.Linear(global_features, 256)
        
    def forward(self, game_state):
        # Encode all cards with context
        all_cards = game_state.get_all_cards()
        card_embeds = self.card_state_encoder(all_cards, game_state)
        
        # Global state (energy, turn, etc.)
        global_embed = self.global_encoder(game_state.get_global_features())
        
        # Combine
        state_embed = torch.cat([card_embeds.mean(0), global_embed])
        return state_embed
```

## 3. World Model

**Just the essentials:**

```python
class WorldModel(nn.Module):
    def __init__(self):
        self.state_encoder = StateEncoder()
        
        # Predict next state
        self.transition_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim)
        )
        
        # Predict reward
        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(), 
            nn.Linear(256, 1)
        )
        
        # Predict game value
        self.value_model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        state_embed = self.state_encoder(state)
        
        next_state = self.transition_model(torch.cat([state_embed, action]))
        reward = self.reward_model(torch.cat([state_embed, action]))
        value = self.value_model(state_embed)
        
        return next_state, reward, value
```

## 4. Simple Planning

**Basic MCTS using world model:**

```python
class SimpleMCTS:
    def __init__(self, world_model):
        self.world_model = world_model
        
    def search(self, state, num_sims=100):
        for _ in range(num_sims):
            # Rollout using world model
            current_state = state
            path_reward = 0
            
            for step in range(5):  # Look ahead 5 steps
                actions = get_legal_actions(current_state)
                action = random.choice(actions)
                
                next_state, reward, _ = self.world_model(current_state, action)
                path_reward += reward
                current_state = next_state
                
                if is_terminal(current_state):
                    break
            
            # Update action values (simplified)
            
        return best_action
```

## 5. Training Loop

**Basic learning:**

```python
def train():
    world_model = WorldModel()
    mcts = SimpleMCTS(world_model)
    replay_buffer = []
    
    for episode in range(1000):
        # Self-play game
        game = PokemonTCGGame()
        episode_data = []
        
        while not game.done():
            state = game.get_state()
            
            # Use MCTS to pick action
            action = mcts.search(state)
            
            # Take action
            next_state, reward = game.step(action)
            episode_data.append((state, action, reward, next_state))
        
        # Add to replay buffer
        replay_buffer.extend(episode_data)
        
        # Train world model
        if len(replay_buffer) > 1000:
            batch = random.sample(replay_buffer, 256)
            train_world_model(world_model, batch)
        
        # Evaluate occasionally
        if episode % 100 == 0:
            win_rate = evaluate_agent(mcts)
            print(f"Episode {episode}, Win Rate: {win_rate}")

def train_world_model(model, batch):
    states, actions, rewards, next_states = zip(*batch)
    
    # Predict
    pred_next_states, pred_rewards, pred_values = model(states, actions)
    
    # Loss
    transition_loss = F.mse_loss(pred_next_states, next_states)
    reward_loss = F.mse_loss(pred_rewards, rewards)
    
    total_loss = transition_loss + reward_loss
    total_loss.backward()
    optimizer.step()
```

## 6. Bootstrap Requirements

**Data:**
- 1000+ game replays (human play or random)
- Basic card database (name, type, hp, attack damage)

**Compute:**
- Single GPU (GTX 1080 sufficient)
- ~10GB storage for replays

**Implementation Time:**
- 2-3 weeks for working prototype
- Start with simplified Pokemon TCG rules

## Success Criteria

1. **Week 1:** World model predicts next states with >60% accuracy
2. **Week 2:** Agent beats random play >70% of time  
3. **Week 3:** Agent shows basic strategic behavior (energy management, targeting)

## That's It

This stripped-down version focuses purely on the learning loop:
1. Collect data
2. Train world model on state transitions  
3. Use world model for planning via MCTS
4. Improve through self-play

Everything else (uncertainty, hierarchical planning, production deployment) can be added later once this core works.