#! /usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight
from std_msgs.msg import Int32
from tf.transformations import euler_from_quaternion

from scipy import spatial

import copy
import math
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5

class CarState:
    ACCEL = 0
    DECEL = 1
    STOP  = 2
    KEEP  = 3


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        ### Member variables
        self.base_waypoints = None
        self.final_waypoints = None
        self.traffic_wp = -1
        self.light_wp = -1
        self.closest_wp = -1
        self.light_state = TrafficLight.UNKNOWN
        self.light_change = False
        self.velocity_cb_state = False 
        self.pose_cb_state = False 
        self.traffic_cb_state = False 
        self.ignore_count = 0
        self.wp_len = 0
        self.car_state = CarState.STOP
        self.waypointsKD = None

        ### Subscribers
        # all waypoints of the track before and after the car
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        ### Publishers
        # publish a fixed number of waypoints ahead of the car starting with the first point ahead of the car
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        ## Main Loop    
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if self.base_waypoints and self.velocity_cb_state and \
                     self.pose_cb_state and self.traffic_cb_state:
                self.closest_wp = self.closest_node()
                self.get_waypoints()
                self.publish_final_waypoints()
            rate.sleep()

    def velocity_cb(self, msg):
        # obtain the current velocity
        self.velocity_cb_state = True 
        self.current_linear_velocity = msg.twist.linear.x
        self.current_angular_velocity = msg.twist.angular.z

    def pose_cb(self, msg):
        # obtain the current pose
        self.pose_cb_state = True 
        self.position = msg.pose.position
        self.orientation = msg.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([
            self.orientation.x, self.orientation.y, self.orientation.z, self.orientation.w])
        self.yaw = yaw

    def get_waypoints(self):
        self.final_waypoints = []
        decel = False
        stop =  False
        tmp_wps = self.base_waypoints.waypoints

        car_x = self.position.x
        car_y = self.position.y
        car_yaw = self.yaw
        car_s = 0
        car_s_list = []
        car_dist_list = []
        car_angle_list = []

        for i in range(self.lookahead_wps):
            wp_index = (self.closest_wp + i + 1) % self.wp_len
            dist, angle = self.get_next_target(tmp_wps[wp_index], car_x, car_y, car_yaw)
            car_s += dist
            car_s_list.append(car_s)
            car_dist_list.append(dist)
            car_angle_list.append(angle)

            next_index = (self.closest_wp + i + 2) % self.wp_len
            car_x = tmp_wps[next_index].pose.pose.position.x
            car_y = tmp_wps[next_index].pose.pose.position.y
            car_yaw = (car_yaw + angle) % (2 * np.pi)

        import warnings
        warnings.simplefilter('ignore', np.RankWarning)
        poly = np.polyfit(car_s_list, car_angle_list, 2)
        self.str_poly = np.poly1d(poly)

        cur_vel = self.current_linear_velocity

        if self.light_wp != -1 and self.ignore_count == 0:
            dist = self.distance(tmp_wps, self.closest_wp, self.light_wp)
            if self.light_change == True and self.light_state == TrafficLight.GREEN \
                and cur_vel < self.speed_limit * 0.2:
                self.car_state = CarState.ACCEL
                self.ignore_count = 8
                rospy.loginfo("phase1:ACCEL-TURN GREEN")
            elif self.light_change == True and self.light_state == TrafficLight.RED \
                and dist > 12.0:
                self.car_state = CarState.DECEL
                self.ignore_count = 2
                rospy.loginfo("phase2:DECEL-TURN YELLOW")
            elif self.light_state == TrafficLight.RED \
                and dist < 3.0:
                self.car_state = CarState.STOP
                self.ignore_count = 0
                rospy.loginfo("phase3:STOP")
            elif self.light_state == TrafficLight.RED \
                and dist < 12.0 and cur_vel < self.speed_limit * 0.6:
                self.car_state = CarState.KEEP
                self.ignore_count = 0
                rospy.loginfo("phase4:KEEP")
            else:
                self.car_state = CarState.DECEL
                self.ignore_count = 0
                rospy.loginfo("phase5:DECEL-CHECK TRAFFIC LIGHT")
        else :
            if self.ignore_count > 0:
                self.ignore_count -= 1
            else:
                self.car_state = CarState.ACCEL
                rospy.loginfo("phase6:ACCEL")

        self.light_change = False
        
        for i in range(self.lookahead_wps):
            wp_index = (self.closest_wp + i + 1) % self.wp_len
            max_l_vel = self.get_waypoint_velocity(self.base_waypoints.waypoints[wp_index])
            if self.car_state == CarState.STOP:
                l_vel = -1.0
            elif self.car_state == CarState.DECEL:
                l_vel = min(max(self.decel_poly(dist - car_s_list[i]), self.speed_limit * 0.3), max_l_vel)
            elif self.car_state == CarState.ACCEL:
                l_vel = max_l_vel
            else:
                l_vel = min(self.current_linear_velocity, self.speed_limit * 0.2)

            calc_angle = self.str_poly(car_s_list[i])
            if calc_angle > 0.0: 
                tmp_angle = min(self.str_poly(car_s_list[i]), car_angle_list[i])
            else:
                tmp_angle = max(self.str_poly(car_s_list[i]), car_angle_list[i])

            if car_dist_list[i] == 0.0:
                car_dist_list[i] = 0.001

            a_vel = tmp_angle * l_vel / car_dist_list[i]

            p = Waypoint()
            p.pose.pose.position.x = self.base_waypoints.waypoints[wp_index].pose.pose.position.x
            p.pose.pose.position.y = self.base_waypoints.waypoints[wp_index].pose.pose.position.y
            p.pose.pose.position.z = self.base_waypoints.waypoints[wp_index].pose.pose.position.z
            p.pose.pose.orientation.x = self.base_waypoints.waypoints[wp_index].pose.pose.orientation.x
            p.pose.pose.orientation.y = self.base_waypoints.waypoints[wp_index].pose.pose.orientation.y
            p.pose.pose.orientation.z = self.base_waypoints.waypoints[wp_index].pose.pose.orientation.z
            p.pose.pose.orientation.w = self.base_waypoints.waypoints[wp_index].pose.pose.orientation.w
            p.twist.twist.linear.x = l_vel
            p.twist.twist.linear.y = 0.
            p.twist.twist.linear.z = 0.
            p.twist.twist.angular.x = 0.
            p.twist.twist.angular.y = 0.
            p.twist.twist.angular.z = a_vel
            self.final_waypoints.append(p)
    
    def publish_final_waypoints(self):
        lane = self.generate_lane();
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        #lane.waypoints = self.final_waypoints
        self.final_waypoints_pub.publish(lane)

    def generate_lane(self):
        lane = Lane()

        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx: farthest_idx]

        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = base_waypoints
        else:
            lane.waypoints = self.decelerate_waypoints(base_waypoints, closest_idx)

        return lane

    def waypoints_cb(self, waypoints):

        self.base_waypoints = waypoints
        # we only need the message once, unsubscribe after first receive
        self.base_waypoints_sub.unregister()

        pointsarr = []
        for i in range(len(self.base_waypoints.waypoints)):
            pointsarr.append([self.base_waypoints.waypoints[i].pose.pose.position.x, 
                              self.base_waypoints.waypoints[i].pose.pose.position.y])
        
        # initialize light KD tree
        self.waypointsKD = spatial.cKDTree(np.asarray(pointsarr), leafsize=10)

        self.wp_len = len(waypoints.waypoints)
        self.speed_limit = waypoints.waypoints[self.wp_len/2].twist.twist.linear.x

        if self.wp_len < LOOKAHEAD_WPS :
            self.lookahead_wps = self.wp_len
        else:
            self.lookahead_wps = LOOKAHEAD_WPS

        rospy.loginfo("Get base_waypoint, way_points = %d, Speed Limit = %f mps.", self.wp_len, self.speed_limit)
        rospy.loginfo("ahead_waypoints = %d.", self.lookahead_wps)

        wpx = [50.0,
               40.0,
               30.0,
               20.0,
               15.0,
               10.0,
               7.0,
               5.0,
               3.0,
               0.0,
               -5.0,
               -50.0]

        wpy = [ self.speed_limit * 0.9,
                self.speed_limit * 0.8,
                self.speed_limit * 0.7,
                self.speed_limit * 0.6,
                self.speed_limit * 0.5,
                self.speed_limit * 0.4,
                self.speed_limit * 0.3,
                self.speed_limit * 0.3,
                self.speed_limit * 0.3,
                0.0,
                -1.0,
                -1.0]

        poly = np.polyfit(np.array(wpx), np.array(wpy), 2)
        self.decel_poly = np.poly1d(poly)
        '''
        rospy.loginfo("wpy[50.] = %f", self.decel_poly(50.))
        rospy.loginfo("wpy[25.] = %f", self.decel_poly(25.))
        rospy.loginfo("wpy[10.] = %f", self.decel_poly(10.))
        rospy.loginfo("wpy[5.]  = %f", self.decel_poly(5.))
        rospy.loginfo("wpy[0.]  = %f", self.decel_poly(0.))
        rospy.loginfo("wpy[-5.] = %f", self.decel_poly(-5.))
        rospy.loginfo("wpy[-50.] = %f", self.decel_poly(-50.))
        '''

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoint_tree.query([x, y],1)[1]

        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]

        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx +1) % len(self.waypoints_2d)

        return closest_idx

    def decelerate_waypoints(self, waypoints, closest_idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx -2, 0)
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.0:
                vel = 0.0

            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)

        return temp

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_wp message. Implement
        str_light = ["RED", "YELLOW", "GREEN", "UNKNOWN", "UNKNOWN"]

        self.traffic_cb_state = True

        self.traffic_wp = int(msg.data)
        if self.traffic_wp > self.wp_len:
            if self.light_state != TrafficLight.UNKNOWN:
                self.light_change = True
            self.light_state = TrafficLight.UNKNOWN
            self.light_wp = -1
        elif self.traffic_wp > 0:
            if self.light_state != TrafficLight.RED:
                self.light_change = True
            self.light_state = TrafficLight.RED
            self.light_wp = self.traffic_wp
        else:
            if self.light_state != TrafficLight.GREEN:
                self.light_change = True
            self.light_state = TrafficLight.GREEN
            self.light_wp = -self.traffic_wp

        #rospy.loginfo("state = %s, light_wp = %d, car_wp = %d.",
        #    str_light[self.light_state], self.light_wp, self.closest_wp)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_next_target(self, waypoint, x, y, yaw):
        # convert to local coordinates
        vx = waypoint.pose.pose.position.x - x
        vy = waypoint.pose.pose.position.y - y
        lx = vx * np.cos(yaw) + vy * np.sin(yaw)
        ly = -vx * np.sin(yaw) + vy * np.cos(yaw)
        dist = math.sqrt(lx * lx + ly * ly)
        angle = np.arctan2(ly, lx)
        return dist, angle

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_linear_velocity(self, waypoint, velocity):
        waypoint.twist.twist.linear.x = velocity

    def set_waypoint_angular_velocity(self, waypoint, angle):
        waypoint.twist.twist.angular.z = angle

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def closest_node(self):

        # current position
        cur_pos_x = self.position.x
        cur_pos_y = self.position.y

        # we can assume the track waypoints are already in a cyclic order
        cur_o = self.orientation
        cur_q = (cur_o.x,cur_o.y,cur_o.z,cur_o.w)
        cur_roll, cur_pitch, cur_yaw = euler_from_quaternion(cur_q)
        
        wp_len = len(self.base_waypoints.waypoints)

        closest_dist, closest_wp = self.waypointsKD.query([cur_pos_x, cur_pos_y], k=1)

        #Check if waypoint is ahead of vehicle
        closest_wp_x = self.base_waypoints.waypoints[closest_wp].pose.pose.position.x
        closest_wp_y = self.base_waypoints.waypoints[closest_wp].pose.pose.position.y
        dist_ahead = ((closest_wp_x - cur_pos_x)* math.cos(cur_yaw) +
                  (closest_wp_y - cur_pos_y)* math.sin(cur_yaw)) > 0.0

        if not dist_ahead:
            closest_wp = (closest_wp + 1) % wp_len

        return closest_wp

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
