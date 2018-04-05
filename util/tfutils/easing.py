"""Module with some easing functions"""
import tensorflow as tf
import numpy as np


def interpolate_linear( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_in_quad"):
        t = tf.clip_by_value(current_time/duration, 0, 1)
        return tf.to_float(change_value*t + start_value)


def ease_in_quad( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_in_quad"):
        t = tf.clip_by_value(current_time/duration, 0, 1)
        return tf.to_float(change_value*t*t + start_value)


def ease_out_quad( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_out_quad"):
        t = tf.clip_by_value(current_time/duration, 0, 1)
        return tf.to_float(-change_value*t*(t-2) + start_value)


def ease_in_out_quad( current_time, start_value, change_value, duration ):
    """
    current_time: float or Tensor
        The current time

    start_value: float or Tensor
        The start value

    change_value: float or Tensor
        The value change of the duration. The final value is start_value + change_value

    duration: float or Tensor
        The duration

    Returns the value for the current time
    """
    with tf.name_scope("ease_in_out_quad"):
        d_2 = 0.5*duration
        c_2 = 0.5*change_value
        return tf.cond( current_time/duration < 0.5, 
                lambda:ease_in_quad(current_time, start_value, c_2, d_2), 
                lambda:ease_out_quad(current_time-d_2, start_value+c_2, c_2, d_2) )


