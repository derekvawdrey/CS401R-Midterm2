# Assignment instructions

This assignment will be less about coding the solution, and more about understanding rewarding of a bot. Your job is to learn how to structure the rewards in order to enable the DQN (Deep Q-Network) to learn how to play the game properly (the type of game will be discussed in Section 0), and how to survive as long as possible. An optional part of this assignment will be training your own model, but the primary part is observing how specific agents learned based on the rewards given.

All the agents were trained on 3000 steps on a grid size 20x20.

The first section of this assignment is learning how the game works, and how to run the game.

## Section 0: Installation and understanding the game.

- Read through the README.md, install the game, and play it once or twice to understand how the game functions. 

## Section 1.1: Agent 1's

This Agent represents the first agent I implemented for the game, with the observations, and rewards.

Run the following code, and observe how agent 1 moves/interacts with the environment. Try to identify what it learns to do.

```
python main.py --mode agent --dqn-model ./pretrained-models/agent_1.pth
```

### Question 1: What does the agent tend to do in this environment?
### Question 2: In most situations, where is the agent when it dies?

---

## Section 1.2: Agent 1's training process

Agent 1 was trained on the rewards and observations in the file, look at the file, and identify possible issues in how it was awarded or the information this agent was provided.

```
meteor-game/agent_rewards_options/agent_1_reward_options.py
```

### Question 1: Why do you think the agent dies the way it does? And how do you think we can limit those types of deaths?
### Question 2: What do you think would happen, if we enabled ENABLE_POSITION_INFO for the agent?

Hints:

Agent 1 currently does not know where it is in relation to the map. It does not know a border exists, and it does not know how close it is to the border.

## Section 2: Agent 2

**This is important:** Copy the contents of ```meteor-game/agent_reward_options/agent_2_reward_options.py``` and put replace the contents of ```meteor-game/agent_reward_options/training_options.py```.

Then run the below command and observe how agent 2 performs.

```
python main.py --mode agent --dqn-model ./pretrained-models/agent_2.pth
```

### Question 1: How did it perform compared to Agent 1?
### Question 2: Why do you think changing ENABLE_POSITION_INFO on the agent made a difference?

## Section 3: Agent 3, Border avoider

Now we want to create an agent that will avoid going near the border, while also avoiding the number of bombs.

Look at the options we have available to us in the ```meteor-game/agent_reward_options/training_options.py```.

### Question 1: What could we change for the agent to avoid the border and stay near the center?

Look at ```meteor-game/agent_reward_options/agent_3_reward_options.py``` and see if you were thinking what I was thinking.

Now run agent 3 and see what it's behavior is.

```
python main.py --mode agent --dqn-model ./pretrained-models/agent_3.pth
```

### Question 2: Does the agent perform how you thought it would?

## Section 4:


