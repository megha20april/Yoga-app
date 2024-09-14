from apps import db

class HeartRateData(db.Model):
    __tablename__ = 'HeartRateData'

    user_id = db.Column(db.Integer, db.ForeignKey('Users.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('YogaSession.id'), primary_key=True)
    pose_name = db.Column(db.String(64), nullable=False)
    timestamp = db.Column(db.DateTime, primary_key=True)
    heart_rate = db.Column(db.Integer, nullable=False)

    session = db.relationship('YogaSession', backref='heart_rate_data')


class YogaPoseData(db.Model):
    __tablename__ = 'YogaPoseData'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('YogaSession.id'), nullable=False)
    pose_name = db.Column(db.String(64), nullable=False)
    avg_heart_rate = db.Column(db.Integer, nullable=True)
    time_spent = db.Column(db.Float, nullable=False)  # in minutes
    calories_burned = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('Users.id'), nullable=False)

    session = db.relationship('YogaSession', backref='pose_data')


class YogaSession(db.Model):
    __tablename__ = 'YogaSession'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('Users.id'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=True)  # Can be calculated
    total_duration = db.Column(db.Float, nullable=True)  # Sum of time from all poses
    total_calories = db.Column(db.Float, nullable=True)  # Sum of calories from all poses

    user = db.relationship('Users', backref='sessions')

    def calculate_total_duration(self):
        self.total_duration = sum(pose.time_spent for pose in self.pose_data)
        return self.total_duration

    def calculate_total_calories(self):
        self.total_calories = sum(pose.calories_burned for pose in self.pose_data)
        return self.total_calories

