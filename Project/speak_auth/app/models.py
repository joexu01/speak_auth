from datetime import datetime

from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

from flask_login import UserMixin, AnonymousUserMixin
from flask import current_app

from app.exceptions import ValidationError

from . import db, login_manager, user_img
from .assist_func import random_dtw_number


# 权限常量
class Permission:
    ASSIST_ADMIN = 1
    ADMIN = 2


# 用户角色Model
class Role(db.Model):
    __tablename__ = 'roles'
    id = db.Column(db.Integer, primary_key=True)  # 角色id
    name = db.Column(db.String(64), unique=True)  # 角色名称
    default = db.Column(db.Boolean, default=False, index=True)
    permissions = db.Column(db.Integer)
    users = db.relationship('User', backref='role', lazy='dynamic')

    def __init__(self, **kwargs):
        super(Role, self).__init__(**kwargs)
        if self.permissions is None:
            self.permissions = 0

    @staticmethod
    def insert_roles():
        roles = {
            '用户': [Permission.ASSIST_ADMIN],
            '管理员': [Permission.ASSIST_ADMIN, Permission.ADMIN]
        }
        default_role = '用户'
        for r in roles:
            role = Role.query.filter_by(name=r).first()
            if role is None:
                role = Role(name=r)
            role.reset_permissions()
            for perm in roles[r]:
                role.add_permission(perm)
            role.default = (role.name == default_role)
            db.session.add(role)
        db.session.commit()

    def add_permission(self, perm):
        if not self.has_permission(perm):
            self.permissions += perm

    def remove_permission(self, perm):
        if self.has_permission(perm):
            self.permissions -= perm

    def reset_permissions(self):
        self.permissions = 0

    def has_permission(self, perm):
        return self.permissions & perm == perm
    # has_permission 方法用位与运算符检查组合权限是否包含指定的单独权限

    def __repr__(self):
        return '<Role %r>' % self.name


# 用户Model
class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), unique=True, index=True)
    name = db.Column(db.String(64))
    confirmed = db.Column(db.Boolean, default=False)
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    password_hash = db.Column(db.String(128))
    avatar = db.Column(db.String(128), default='default.jpg')
    mfcc_csv = db.Column(db.String(128))
    dtw_mfcc = db.Column(db.String(128))
    dtw_number = db.Column(db.String(6))

    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        if self.role is None:
            if self.email == current_app.config['SPEAK_AUTH_ADMIN']:
                self.role = Role.query.filter_by(name='管理员').first()
            if self.role is None:
                self.role = Role.query.filter_by(default=True).first()
        self.dtw_number = random_dtw_number()

    @property
    def password(self):
        raise AttributeError('password是一个不可访问的属性')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    # 生成确认邮箱的令牌
    def generate_confirmation_token(self, expiration=3600):
        serial = Serializer(current_app.config['SECRET_KEY'], expiration)
        return serial.dumps({'confirm': self.id}).decode('utf-8')

    # 确认邮箱令牌
    def confirm(self, token):
        serial = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = serial.loads(token.encode('utf-8'))
        except:
            return False
        if data.get('confirm') != self.id:
            return False
        self.confirmed = True
        db.session.add(self)
        return True

    # 生成重置令牌
    def generate_reset_token(self, expiration=3600):
        serial = Serializer(current_app.config['SECRET_KEY'], expiration)
        return serial.dumps({'reset': self.id}).decode('utf-8')

    # 重置密码
    @staticmethod
    def reset_password(token, new_password):
        serial = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = serial.loads(token.encode('utf-8'))
        except:
            return False
        user = User.query.get(data.get('reset'))
        if user is None:
            return False
        user.password = new_password
        db.session.add(user)
        return True

    def generate_email_change_token(self, new_email, expiration=3600):
        serial = Serializer(current_app.config['SECRET_KEY'], expiration)
        return serial.dumps(
            {'change_email': self.id, 'new_email': new_email}).decode('utf-8')

    def change_email(self, token):
        serial = Serializer(current_app.config['SECRET_KEY'])
        try:
            data = serial.loads(token.encode('utf-8'))
        except:
            return False
        if data.get('change_email') != self.id:
            return False
        new_email = data.get('new_email')
        if new_email is None:
            return False
        if self.query.filter_by(email=new_email).first() is not None:
            return False
        self.email = new_email
        db.session.add(self)
        return True

    # 权限确认方法
    def can(self, perm):
        return self.role is not None and self.role.has_permission(perm)

    def is_administrator(self):
        return self.can(Permission.ADMIN)

    def delete(self):
        if self.is_administrator():
            return False

    def __repr__(self):
        return '<User %r>' % self.name


class Person(db.Model):
    __tablename__ = 'persons'

    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.String(64), unique=True, index=True)  # 唯一识别码
    person_name = db.Column(db.String(64))
    dtw_number = db.Column(db.String(6))
    dtw_mfcc = db.Column(db.String(64))
    avatar = db.Column(db.String(64), default='default.jpg')
    login_records = db.relationship('LoginRecord', backref='person', lazy='dynamic')

    def __init__(self, *args, **kwargs):
        super(Person, self).__init__(*args, **kwargs)

    @staticmethod
    def create_from_json(json_post):
        uid = json_post.get('uid')
        if uid is None or uid == '':
            raise ValidationError('uid格式不正确')
        if Person.query.filter_by(uid=uid).first() is not None:
            raise ValidationError('uid已被使用，请更换uid')
        return Person(uid=uid, person_name=json_post.get('person_name'),
                      dtw_number=random_dtw_number())

    def to_json(self):
        json_person = {
            'uid': self.uid,
            'person_name': self.person_name,
            'dtw_number': self.dtw_number,
            'avatar_url': user_img.url(self.avatar),
        }
        return json_person

    def __repr__(self):
        return 'Person %r' % self.person_name


class LoginRecord(db.Model):
    __tablename__ = 'login_records'

    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey('persons.id'))
    dtw_result = db.Column(db.Boolean, nullable=False)
    svm_result = db.Column(db.Boolean, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, *args, **kwargs):
        super(LoginRecord, self).__init__(*args, **kwargs)

    def __repr__(self):
        return 'LoginRecord %d' % self.id


# 未登录用户模型
class AnonymousUser(AnonymousUserMixin):
    def can(self, permissions):
        return False

    def is_administrator(self):
        return False


# 加载用户信息给Flask_Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
