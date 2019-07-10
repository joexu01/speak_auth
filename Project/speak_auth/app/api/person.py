# -*- coding: UTF-8 -*-

import os

from flask import jsonify, request, url_for, current_app
from .. import db, audio
from ..assist_func import random_string
from ..models import Person, LoginRecord
from . import api
from .errors import bad_request
from .threading_functions import extract_mfcc
from .dtw_auth import auth_pipeline

# # 新建身份
# @api.route('/new_person/step-1', methods=['GET'])
# def new_person_1st_step():
#
#     person = Person.create_from_json(request.json)
#     db.session.add(person)
#     db.session.commit()
#     return jsonify(person.to_json()), 201, \
#         {'2nd_step_url': url_for('api.new_person_2nd_step', uid=person.uid)}


@api.route('/new_person/step-1', methods=['GET'])
def new_person_1st_step():
    person_dict = {
        'uid': request.args.get('uid'),
        'person_name': request.args.get('person_name')
    }
    if Person.query.filter_by(uid=request.args.get('uid')).first() is not None:
        return bad_request('此UID已经被占用，请更换uid后重试')
    person = Person.create_from_json(person_dict)
    db.session.add(person)
    db.session.commit()
    return jsonify(person.to_json()), 201, \
        {'2nd_step_url': url_for('api.new_person_2nd_step', uid=person.uid)}


@api.route('/new_person/step-2/<uid>', methods=['POST'])
def new_person_2nd_step(uid):
    person = Person.query.filter_by(uid=uid).first()
    if person is None:
        return bad_request('没有此人的身份信息，请先创建身份')
    if request.files:
        audio_suffix = os.path.splitext(request.files['audio_file'].filename)[1]
        audio_filename = random_string() + audio_suffix
        audio.save(request.files['audio_file'], name=audio_filename)
        extract_mfcc(user_id=person.uid,
                     audio_dir=current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename,
                     save_to=current_app.config['UPLOADED_PERSON_DEST'])
        person.dtw_mfcc = str(person.uid) + '.csv'
        db.session.add(person)
        db.session.commit()
        return jsonify({'status': "成功提取MFCCs!"}), 200
    return bad_request('没有文件上传到服务器')


@api.route('/authentication/step-1/<uid>', methods=['GET'])
def auth_1st_step(uid):
    person = Person.query.filter_by(uid=uid).first()
    if person is None:
        return bad_request('没有此人的身份信息，请先创建身份')
    return jsonify(person.to_json()), 200, \
           {'2nd_step_url': url_for('api.new_person_2nd_step', uid=person.uid)}


@api.route('/authentication/step-2/<uid>', methods=['POST'])
def auth_2nd_step(uid):
    person = Person.query.filter_by(uid=uid).first()
    if person is None:
        return bad_request('没有此人的身份信息，请先创建身份')
    if request.files['audio_file']:
        audio_suffix = os.path.splitext(request.files['audio_file'].filename)[1]
        audio_filename = random_string() + audio_suffix
        audio.save(request.files['audio_file'], name=audio_filename)
        dtw_result, real_man_result = auth_pipeline(dtw_features_path=
                                                    current_app.config['UPLOADED_PERSON_DEST'] + '/' + person.dtw_mfcc,
                                                    audio_file_path=
                                                    current_app.config['UPLOADED_AUDIO_DEST'] + '/' + audio_filename,
                                                    common_data_path=
                                                    current_app.config['GMM_COMMON_DATA'],
                                                    throat_data_path=
                                                    current_app.config['GMM_THROAT_DATA'])
        login_record = LoginRecord(person_id=person.id,
                                   dtw_result=dtw_result,
                                   svm_result=real_man_result)
        db.session.add(login_record)
        db.session.commit()
        if dtw_result and real_man_result:
            return jsonify({'status': "恭喜！验证通过", 'result': 'True'}), 200
        return jsonify({'status': "抱歉！验证未通过", 'result': 'False'}), 200
