from gym.envs.registration import register

register(
    id='gym_storehouse-v0',
    entry_point='gym_storehouse.envs:StoreHouse',
)