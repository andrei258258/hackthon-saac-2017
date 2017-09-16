import sys
import gym
from gym import wrappers

from top_agents import MinistryOfSillyWalkAgent as Agent

env = gym.make('BipedalWalker-v2')
# You provide the directory to write to (can be an existing
# directory, including one with existing data -- all monitor files
# will be namespaced). You can also dump to a tempdir if you'd
# like: tempfile.mkdtemp().
#outdir = 'C:\\Users\\Alex\\Desktop\\hackthon-saac-2017'
#env = wrappers.Monitor(env, directory=outdir, force=True)
env.seed(0)
agent = Agent(env.action_space)

episode_count = 20
reward = 0
done = False

for i in range(episode_count):
  ob = env.reset()
  while True:
      action = agent.act(ob, reward, done)
      ob, reward, done, _ = env.step(action)
      env.render()
      if done:
          break
      # Note there's no env.render() here. But the environment still can open window and
      # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
      # Video is not recorded every episode, see capped_cubic_video_schedule for details.
print(ob)
# Close the env and write monitor result info to disk
env.close()

# Upload to the scoreboard. We could also do this from another
# process if we wanted.
#gym.upload(outdir)
