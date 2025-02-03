import random
import pandas as pd
import math
import methods as md
from importlib import reload
md = reload(md)

drivers = pd.read_excel('drivers.xlsx')
riders = pd.read_excel('riders.xlsx')

rider_waiting_time = 0 # total waiting time for riders
abandonments = 0
earnings = 0 # total earnings for drivers
number_of_pickups = [] # number of pickups per driver
driver_rest_time = 0 # total time between pickups for all drivers
driver_total_time = 0 # total working hours of all drivers
current_driver_id = 1
# time when drivers go online
driver_online_queue = [(current_driver_id, md.generate_random('exp',3), md.generate_location())] # (id,time,location)
drivers_available = {} # id:(location)
number_of_pickups.append(0)
# time when drivers go offline
driver_offline = [] # (id,time)
current_rider_id = 1
# time when riders request a ride
ride_requested = [(current_rider_id, md.generate_random('exp',30), md.generate_location(), md.generate_location())] # (id,time,origin, destination)
# time when riders run out of patience and cancel a ride
rider_offline = [] # (id,time)
riders_waiting = {} # id:(location, destination)
# matched rides
ride_complete = [] # (id, end time, origin, destination, driver)
T = 3 # Set termination time (by the minute)
t = 0 # Current time (minute 0)

while t < T:
    if len(driver_offline) > 0:
        minimum = min(driver_online_queue[0][1], driver_offline[0][1], ride_requested[0][1])
    else:
        minimum = min(driver_online_queue[0][1], ride_requested[0][1])
    if len(rider_offline) > 0:
        minimum = min(minimum, rider_offline[0][1])
    if len(ride_complete) > 0:
        minimum = min(minimum, ride_complete[0][1])
    t = minimum
    if driver_online_queue[0][1] == t:
        # add driver to available dictionary
        drivers_available[current_driver_id] = [driver_online_queue[0][2]]
        # generate time driver goes offline
        new_off = (current_driver_id, t + md.generate_random('unif',3) + 5)
        driver_offline = md.add_to_list(new_off, driver_offline)
        # look for riders
        rider = md.find_closest_available(driver_online_queue[0][2], riders_waiting)
        # rider found
        if rider:
            # generate time taken to travel to rider
            ride_start = t + md.generate_trip_time(driver_online_queue[0][2], riders_waiting[rider][0])
            # generate time taken to reach destination
            new_ride = (rider, ride_start + md.generate_trip_time(riders_waiting[rider][0], riders_waiting[rider][1]), riders_waiting[rider][0], riders_waiting[rider][1], driver_online_queue[0][0])
            ride_complete = md.add_to_list(new_ride, ride_complete)
            del riders_waiting[rider]
        else:
            # add driver back to available
            drivers_available[current_driver_id] = [driver_online_queue[0][2]]
        number_of_pickups.append(0)
        driver_online_queue.pop(0)
        # generate new driver to online queue
        current_driver_id += 1
        new_on = (current_driver_id,t + md.generate_random('exp',3), md.generate_location())
        driver_online_queue = md.add_to_list(new_on, driver_online_queue)
    if len(driver_offline) > 0:
        if driver_offline[0][1] == t:
            # delete driver from available dictionary
            try:
                del drivers_available[driver_offline[0][0]]
            except:
                pass
            driver_offline.pop(0)
    # a rider requests a ride
    if ride_requested[0][1] == t:
        # find driver
        driver = md.find_closest_available(ride_requested[0][2], drivers_available)
        if driver:
            # driver found immediately, generate time taken to travel to rider
            ride_start = t + md.generate_trip_time(drivers_available[driver][0], ride_requested[0][2])
            # generate time taken to reach destination
            new_ride = (ride_requested[0][0], ride_start + md.generate_trip_time(ride_requested[0][2], ride_requested[0][3]), ride_requested[0][2], ride_requested[0][3], driver)
            ride_complete = md.add_to_list(new_ride, ride_complete)
            del drivers_available[driver]
        else:
            # generate and add offline time if driver not found (patience)
            rider_offline = md.add_to_list((ride_requested[0][0], t + md.generate_random('exp',5)), rider_offline)
            # add rider to waiting dictionary
            riders_waiting[ride_requested[0][0]] = (ride_requested[0][2], ride_requested[0][3])
        ride_requested.pop(0)
        # generate new rider to online queue
        current_rider_id += 1
        ride_requested = md.add_to_list((current_rider_id, t + md.generate_random('exp',30), md.generate_location(), md.generate_location()), ride_requested)
    if len(rider_offline) > 0:
        if rider_offline[0][1] == t:
            try:
                del riders_waiting[rider_offline[0][0]]
                abandonments += 1
            # not in waiting dictionary -- already got picked up
            except:
                pass
            rider_offline.pop(0)
    if len(ride_complete) > 0:
        if ride_complete[0][1] == t:
            # calculate profits of driver (revenue - fuel)
            earnings = earnings + md.calculate_trip(ride_complete[0][2], ride_complete[0][3]) - md.calculate_petrol(ride_complete[0][2], ride_complete[0][3])
            number_of_pickups[current_driver_id - 1] += 1
            # check if driver has ended work shift
            for i in driver_offline:
                if i[0] == ride_complete[0][4]:
                    # his shift has not ended
                    if i[1] > t:
                        # look for riders, driver is free
                        rider = md.find_closest_available(ride_complete[0][3], riders_waiting)
                        if rider:
                            ride_start = t + md.generate_trip_time(ride_complete[0][3], riders_waiting[rider][0])
                            new_ride = (rider, ride_start + md.generate_trip_time(riders_waiting[rider][0], riders_waiting[rider][1]), riders_waiting[rider][0], riders_waiting[rider][1], ride_complete[0][3])
                            ride_complete = md.add_to_list(new_ride, ride_complete)
                            del riders_waiting[rider]
                        else:
                            # add driver back to available
                            drivers_available[ride_complete[0][4]] = [ride_complete[0][3]]
                    else:
                        # record his statistics
                        pass
                    break
            # remove rider from system
            ride_complete.pop(0)