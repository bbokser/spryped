# spryped

I am developing a bipedal robot featuring my custom made QDD actuator, SpryDrive. Compare the "chickenwalker" design to Agility Robotics' Cassie, as well as the Blackbird bipedal robot. The focus of my design is to move the CoM as far upward as possible (to better match the SLIP model) while also reducing limb inertia and optimizing for low weight, profile, and simplicity. To achieve this, I have located the ankle and toe actuators in the tibiotarsus member and set the ratio of the lengths of the tibiotarsus member to the tarsometatarsus member as 2:5 (whereas traditionally a roughly 1:1 ratio is used). "Toe" (or foot) actuation is attained with the use of a four-bar linkage mechanism, although I am also considering a belt drive. It is unlikely that I will include yaw actuation in the first iteration, as it is almost exclusively used for turning rather than the acts of balancing and walking.

This repo is being built for use with PyBullet. 

I am currently learning to use methods such as whole body operational space control for the balancing controller. 
