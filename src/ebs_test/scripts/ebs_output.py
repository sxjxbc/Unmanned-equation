#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exclusive transport for the existing steering/stop protocol only."""
import socket


def validate_output_request(enabled, mode, profile):
    expected = 'original_protocol_test' if enabled else 'shadow_only'
    if mode != expected:
        raise RuntimeError('Output request and output_mode disagree: %s' % mode)
    if enabled and profile != 'baseline_shadow':
        raise RuntimeError('Vehicle output requires the baseline profile')


def exclusive_client_class(base):
    class ExclusiveCANUDPClient(base):
        def __init__(self, destination, destination_port, source_port):
            base.__init__(self)
            self.dest_ip = destination
            self.dest_port = int(destination_port)
            self.self_port = int(source_port)
            self.sent_frames = 0
            self.sent_stop_frames = 0
            self.last_send_ok = None

        def init(self):
            # Do not set SO_REUSEADDR/PORT: an existing original sender must
            # prevent this sender from opening the same channel, and vice versa.
            self.close()
            candidate = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                candidate.bind(('', self.self_port))
                self.addr_to = (self.dest_ip, self.dest_port)
                self.sock = candidate
                return True
            except Exception:
                candidate.close()
                raise

        def send_steering_frame(self, angle, stop=False):
            # Keep the actual straight-program codec, including raw=2000.
            self.last_send_ok = base.send_steering_frame(self, angle, stop=stop)
            if self.last_send_ok:
                self.sent_frames += 1
                self.sent_stop_frames += int(bool(stop))
            return self.last_send_ok

    return ExclusiveCANUDPClient
