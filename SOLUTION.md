# Agent 1

Agent 1's problem is that it hugs the outside ring of the map. Once it is on the outside ring it will die if two meteors are clustered together by it, or if the agent moves to the corner.

## Section 1.1

Agent 1 is an agent that does not know where it is in relation to the center of the map.

### Question 1 answer:

Agent 1 tends to move to the border of the map.

### Question 2 answer:

Agent 1's death is gauanteed once it reaches the border. Generally it does a good job at avoiding bombs at the beginning of the game, but once it reaches the border, it dies

## Section 1.2

### Question 1 answer:

The agent is not penalized for being near the border, this is one possible answer. There is no real answer, more of a through exercise.

### Question 2 answer:

The agent doesn't know where it is relative to the center or border yet, so maybe giving it information on where the border is will help it avoid bombs

---

## Section 2

Agent 2 represents an agent with more understanding of its environment, specifically its coordinates and relative position to the center and border. 

### Question 1 answer:

Agent 2 performed much better than Agent 1, it survived for a longer time on average compared to Agent 2. 

### Question 2 answer:

Agent 2 now knows that there is a border, and that it can't move past the border. Therefore it knows how to navigate around the border.

## Section 3

Agent 3 is incentivized to be near the center and be away from the border.

### Question 1 answer:

You could do a couple of things:
- Reward it for being near the center
- Penalize it for going near the border

Or any other number of things.

### Question 2 answer:

Very subjective, no real answer

At the end of this, you should have a general idea of what is important to give an agent, and why rewarding/phrasing the problem is important.