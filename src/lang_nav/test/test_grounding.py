import os

import pytest

from lang_nav.grounding import GroundingResult, KeywordGrounder, LandmarkMap

LM = os.path.join(os.path.dirname(__file__), '..', 'config', 'landmarks_tb3_world.yaml')


@pytest.fixture
def lmap():
    return LandmarkMap.from_yaml(LM)


@pytest.mark.parametrize('cmd,name', [
    ('go to the north end', 'north_end'),
    ('Head over to the south east corner please', 'southeast_corner'),
    ('return home', 'home'),
    ('drive next to the west wall', 'west_wall'),
])
def test_keyword_landmarks(lmap, cmd, name):
    r = KeywordGrounder(lmap).ground(cmd)
    assert r.kind == 'landmark' and r.landmark == name


def test_keyword_relative(lmap):
    r = KeywordGrounder(lmap).ground('move forward 1.5 m then turn left')
    assert r.kind == 'relative' and r.forward_m == 1.5 and r.turn_deg == 90.0


def test_keyword_none(lmap):
    assert KeywordGrounder(lmap).ground('make me a sandwich').kind == 'none'


def test_validate(lmap):
    assert GroundingResult('landmark', landmark='nope').validate(lmap).kind == 'none'
    assert GroundingResult('image_point', pixel_u=5, pixel_v=5).validate(lmap).kind == 'none'
    assert GroundingResult('image_point', pixel_u=5, pixel_v=5).validate(lmap, (640, 480)).kind == 'image_point'
    assert GroundingResult('image_point', pixel_u=700, pixel_v=5).validate(lmap, (640, 480)).kind == 'none'
    assert GroundingResult('relative', forward_m=9).validate(lmap).kind == 'none'


def test_back_home_is_landmark(lmap):
    r = KeywordGrounder(lmap).ground('take me back home')
    assert r.kind == 'landmark' and r.landmark == 'home'
    r = KeywordGrounder(lmap).ground('go back 0.5 m')
    assert r.kind == 'relative' and r.forward_m == -0.5


def test_debug_pixel(lmap):
    r = KeywordGrounder(lmap).ground('@pixel 160 113')
    assert (r.kind, r.pixel_u, r.pixel_v) == ('image_point', 160, 113)
