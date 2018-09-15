# Utility script to show Gym Environments
from gym import envs

env_names = [spec.id for spec in envs.registry.all()]

for names in sorted(env_names):
    print(names)

print('\nTOTAL ENVIRONMENTS: ' + str(len(env_names)))
