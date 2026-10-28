#ifndef NMPC_H_
#define NMPC_H_
 
#include <casadi/casadi.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
// #include "master/motion/vicon_error_net_infer.h"


// namespace trobot {
// namespace master {
// namespace motion {

struct MpcData {
    std::vector<double> local_timestamp;
    std::vector<std::vector<double>> lbx_list;
    std::vector<std::vector<double>> ubx_list;
    std::vector<std::vector<double>> lbg_list;
    std::vector<std::vector<double>> ubg_list;
    std::vector<std::vector<double>> m_initial_guess_list;
    std::vector<std::vector<double>> parameters_list;

    std::vector<std::vector<double>> dvx_list;
    std::vector<std::vector<double>> dvy_list;
    std::vector<std::vector<double>> do1_list;
    std::vector<std::vector<double>> do2_list;
    std::vector<std::vector<double>> do3_list;
    std::vector<std::vector<double>> do4_list;
    std::vector<std::vector<double>> do5_list;

    std::vector<int8_t> is_success;
    std::vector<std::vector<double>> end_pos;

    std::vector<double> mpc_cost_time;

    void clear_all() {
        local_timestamp.clear();
        lbx_list.clear();
        ubx_list.clear();
        lbg_list.clear();
        ubg_list.clear();
        m_initial_guess_list.clear();
        parameters_list.clear();

        dvx_list.clear();
        dvy_list.clear();
        do1_list.clear();
        do2_list.clear();
        do3_list.clear();
        do4_list.clear();
        do5_list.clear();

        is_success.clear();
        end_pos.clear();

        mpc_cost_time.clear();
    }
};

#define NMPC_API __attribute__((visibility("default")))
class NMPC_API NMPC
{
private:
    /* data */
    // MPC参数
    int m_predict_step; //一次采样间隔预测的步数
    int n_states; //共有n个状态-----3
    int n_controls; //共有n个控制输入-----2
    std::vector<double> m_initial_guess; //初始猜测解
    //符号定义
    casadi::Function m_solver; //求解器
    casadi::Function m_fk_solver;
    casadi::Function m_predict_fun; //预测函数
    std::map<std::string, casadi::DM> m_res; //求解结果
    std::map<std::string, casadi::DM> m_args; //求解参数
    //求解得到的控制量
    std::vector<double> m_control_command; 
    // 上一次被接受解的首段加速度，用于约束连续两次重规划之间的加速度跳变。
    std::vector<double> m_previous_acceleration;
    int solve_number;
    float motor5_limit_low_;
    float motor5_limit_high_;
    
    casadi::SX GetEndPos(casadi::SX thetas);
    // double hit_speed;    // x 方向的击球速度
    // double hit_speed_z;  // z 方向的击球速度
    // double hit_angle;  // 控制击球时刻球拍的角度
    // double hit_yaw;
    int inference_count;
    int hit_type; // 击球类型，0: 未确定, 1: 前场球 2: 正常球 3： 高远球
    double racket_len;

    double accuracy_threshold; // MPC求解结果的精度阈值
    double motor2_angle1, motor2_angle2;
    double motor3_angle1, motor3_angle2;

    casadi::DM m_prev_lam_x; // 上一次的lam_x 
    casadi::DM m_prev_lam_g; // 上一次的lam_g

    MpcData mpc_data; //存储MPC相关数据
    std::mutex mtx_mpc_states;
    std::vector<double> mpc_states;

    bool predicted_v_xy_valid = false; // 预测的xy方向速度是否有效
    double predicted_vx_at_hit = 0.0f; // 预测的击球时刻的x方向速度
    double predicted_vy_at_hit = 0.0f; // 预测的击球时刻的y方向速度

    // 根据击球位置的y坐标动态限位1号、4号电机
    // void angle_hard_limit(double init_angle, double &motor_angle1, double &motor_angle2, double angle_limit);
    // void dynamic_adjust_constraints(double target_y, double &motor1_angle1, double &motor1_angle2, double &motor4_angle1, double &motor4_angle2);
    void limit_angle(double init_angle, double &motor_angle1, double &motor_angle2, double limit);
    void dynamic_adjust_constraints(double target_y, double &motor_angle1, double &motor_angle2);
    
    // error net
    // std::shared_ptr<ViconErrorNetInfer> error_net_infer_;
    // double error_net_predict_result_[3] = {0.0, 0.0, 0.0}; // error net预测的结果，分别为x、y、z方向的误差
    
 
public:
    NMPC(int predict_step);
    ~NMPC();
    //定义求解器
    void set_my_nmpc_solver();
    void set_my_nmpc_solver_dll();
    //优化求解
    // std::vector<double>  opti_solution(std::vector<double> current_states,
    //                       std::vector<double> desired_states,
    //                       double times);
    std::vector<double> opti_solution( std::vector<double> current_states, 
                                        std::vector<double> desired_states, 
                                        std::vector<double> times,
                                        bool& success, bool& acc_success,
                                        double hit_spd, double hit_yaw, double hit_pitch, double hit_speed_z, double local_timestamp);
    // 获取最优加速度向量，按 [step0(7), step1(7), ...] 排列。
    std::vector<double> get_controls();
    void SetGuss();
    void setInitGuss(std::vector<double> ps);
    void setInitGuss_Order(std::vector<double> ps);
    std::vector<double> GetEndPos2(std::vector<double> thetas, std::vector<double> target_states, bool isprint);

    void LogMpcData(const char* label, std::string timestamp_str, long long ts);
    void SetMpcStates(const std::vector<double> &cur_states);
    std::vector<double> GetMpcStates();
};
 
// }
// }
// }
 
#endif
