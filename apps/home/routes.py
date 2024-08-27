# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request, redirect, url_for
from flask_login import login_required
from jinja2 import TemplateNotFound

from apps.config import API_GENERATOR

@blueprint.route('/index')
@login_required
def index():
    pose_data = [
    {"name": "Cobra Pose (Bhujangasana)", "level": "Beginner", "color": "#00fc7e", "pose_key": "Cobra", "next_pose_key": "Chair"},
    {"name": "Chair Pose (...)", "level": "Beginner", "color": "#00fc7e", "pose_key": "Chair", "next_pose_key": "Tree"},
    {"name": "Tree Pose (Vrksasana)", "level": "Intermediate", "color": "#63e5ff", "pose_key": "Tree", "next_pose_key": "Done"},
    # Add more poses as needed
    ]
    return render_template('home/index.html', segment='index', API_GENERATOR=len(API_GENERATOR), show_sideBar=True, poses=pose_data)

@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        if segment == 'interface':
            return redirect(url_for('video_feed_blueprint.analyze', pose_index=0))        
        return render_template("home/" + template, segment=segment, API_GENERATOR=len(API_GENERATOR), show_sideBar=True)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None
