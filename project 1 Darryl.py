import random
import pandas as pd
import math
import methods as md
from importlib import reload
md = reload(md)

drivers = pd.read_excel('drivers.xlsx')
riders = pd.read_excel('riders.xlsx')

rider_waiting_time = [] # waiting time for each rider
abandonments = 0
earnings = [] # earnings for each driver
number_of_pickups = [] # number of pickups per driver
driver_work_time = [] # total driving time for each driver (initial location to rider + rider to destination)
driver_total_time = [] # total working hours of each driver
shift_end_times = [] # time when drivers end their shift
current_driver_id = 1
# time when drivers start their shift
driver_online_queue = [(current_driver_id, md.generate_random('exp',3), md.generate_location())] # (id,time,location)
drivers_available = {} # id:(location)
number_of_pickups.append(0)
driver_work_time.append(0)
earnings.append(0)
# time when drivers end their shift
driver_offline = [] # (id,time)
current_rider_id = 1
# time when riders request a ride
ride_requested = [(current_rider_id, md.generate_random('exp',30), md.generate_location(), md.generate_location())] # (id,time,origin, destination)
# time when riders run out of patience and cancel a ride
rider_offline = [] # (id,time)
riders_waiting = {} # id:(location, destination)
# matched rides
ride_complete = [] # (id, end time, origin, destination, driver)
ride_info = []
T = 10 # Set termination time
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
        # generate time driver goes offline
        off = md.generate_random('unif',3) + 5
        new_off = (current_driver_id, t + off)
        driver_total_time.append(off)
        shift_end_times.append(t + off)
        driver_offline = md.add_to_list(new_off, driver_offline)
        # look for riders
        rider = md.find_closest_available(driver_online_queue[0][2], riders_waiting)
        # rider found
        if rider:
            number_of_pickups[current_driver_id - 1] += 1
            potential_profit = md.calculate_trip(riders_waiting[rider][0],riders_waiting[rider][1]) - md.calculate_petrol(driver_online_queue[0][2], riders_waiting[rider][0])
            earnings[current_driver_id - 1] += potential_profit
            # earnings[current_driver_id - 1] -= md.calculate_petrol(driver_online_queue[0][2], riders_waiting[rider][0])
            # generate time taken to travel to rider
            time_to_rider = md.generate_trip_time(driver_online_queue[0][2], riders_waiting[rider][0])
            ride_start = t + time_to_rider
            # generate time taken to reach destination
            time_to_dest = md.generate_trip_time(riders_waiting[rider][0], riders_waiting[rider][1])
            new_ride = (rider, ride_start + time_to_dest, riders_waiting[rider][0], riders_waiting[rider][1], driver_online_queue[0][0])
            driver_work_time[current_driver_id - 1] += time_to_rider + time_to_dest
            ride_complete = md.add_to_list(new_ride, ride_complete)
            del riders_waiting[rider]
            rider_waiting_time[rider - 1] = t - rider_waiting_time[rider - 1]
        else:
            # add driver back to available
            drivers_available[current_driver_id] = [driver_online_queue[0][2]]
        driver_online_queue.pop(0)
        # generate new driver to online queue
        current_driver_id += 1
        new_on = (current_driver_id,t + md.generate_random('exp',3), md.generate_location())
        driver_online_queue = md.add_to_list(new_on, driver_online_queue)
        number_of_pickups.append(0)
        earnings.append(0)
        driver_work_time.append(0)
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
        rider_waiting_time.append(t)
        # find driver
        driver = md.find_closest_available(ride_requested[0][2], drivers_available)
        if driver:
            number_of_pickups[driver - 1] += 1
            potential_profit = md.calculate_trip(ride_requested[0][2], ride_requested[0][3]) - md.calculate_petrol(drivers_available[driver][0], ride_requested[0][2])
            earnings[driver - 1] += potential_profit
            # earnings[driver - 1] -= md.calculate_petrol(drivers_available[driver][0], ride_requested[0][2])
            # driver found immediately, generate time taken to travel to rider
            time_to_rider = md.generate_trip_time(drivers_available[driver][0], ride_requested[0][2])
            ride_start = t + time_to_rider
            # generate time taken to reach destination
            time_to_dest = md.generate_trip_time(ride_requested[0][2], ride_requested[0][3])
            new_ride = (ride_requested[0][0], ride_start + time_to_dest, ride_requested[0][2], ride_requested[0][3], driver)
            driver_work_time[driver - 1] += time_to_rider + time_to_dest
            ride_complete = md.add_to_list(new_ride, ride_complete)
            del drivers_available[driver]
            rider_waiting_time[-1] = 0
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
                ride_info.append([rider_offline[0][0],-1,(-1,-1),(-1,-1),-1])
            # not in waiting dictionary -- already got picked up
            except:
                pass
            rider_waiting_time[rider_offline[0][0] - 1] = t - rider_waiting_time[rider_offline[0][0] - 1]
            rider_offline.pop(0)
    if len(ride_complete) > 0:
        if ride_complete[0][1] == t:
            # calculate profits of driver (revenue - fuel)
            earnings[ride_complete[0][4] - 1] += md.calculate_trip(ride_complete[0][2], ride_complete[0][3]) - md.calculate_petrol(ride_complete[0][2], ride_complete[0][3])
            # check if driver has ended work shift
            if ride_complete[0][4] in [x[0] for x in driver_offline]:
                rider = md.find_closest_available(ride_complete[0][3], riders_waiting)
                if rider:
                    number_of_pickups[ride_complete[0][4] - 1] += 1
                    potential_profit = md.calculate_trip(riders_waiting[rider][0], riders_waiting[rider][1]) - md.calculate_petrol(ride_complete[0][3], riders_waiting[rider][0])
                    earnings[ride_complete[0][4] - 1] += potential_profit
                    # earnings[ride_complete[0][4] - 1] -= md.calculate_petrol(ride_complete[0][3], riders_waiting[rider][0])
                    # generate time taken to travel to rider
                    time_to_rider = md.generate_trip_time(ride_complete[0][3], riders_waiting[rider][0])
                    ride_start = t + time_to_rider
                    # generate time taken to reach destination
                    time_to_dest = md.generate_trip_time(riders_waiting[rider][0], riders_waiting[rider][1])
                    new_ride = (rider, ride_start + time_to_dest, riders_waiting[rider][0], riders_waiting[rider][1], ride_complete[0][4])
                    driver_work_time[ride_complete[0][4] - 1] += time_to_rider + time_to_dest
                    ride_complete = md.add_to_list(new_ride, ride_complete)
                    del riders_waiting[rider]
                    rider_waiting_time[rider - 1] = t - rider_waiting_time[rider - 1]
                else:
                    # add driver back to available
                    drivers_available[ride_complete[0][4]] = [ride_complete[0][3]]
                # record statistics
            # driver has ended work shift
            else:
                # record statistics
                pass
            # remove rider from system
            ride_info.append(ride_complete.pop(0))